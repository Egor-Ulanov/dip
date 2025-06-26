from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import re
import time
import os
import smtplib
from email.mime.text import MIMEText
import json


# Глобальные переменные для моделей
review_model = None
review_vectorizer = None
sentiment_model = None
sentiment_vectorizer = None
db = None

# DEBUG и Email функции
DEBUG_CHAT_ID = os.getenv("DEBUG_CHAT_ID", "-4661677635") # ID твоего личного чата или тестовой группы
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def send_debug_message(text):
    if not TELEGRAM_TOKEN or not DEBUG_CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={
            "chat_id": DEBUG_CHAT_ID,
            "text": f"[DEBUG]\n{text}",
            "parse_mode": "Markdown"
        })
        time.sleep(0.3) # не даём отправлять слишком быстро
    except Exception as e:
        print("Ошибка при отправке debug-сообщения:", e)

def send_email(to_email, subject, body):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = os.getenv("EMAIL_USERNAME", "egorulanov908@gmail.com") # поменяй на свою почту или используй ENV
    password = os.getenv("EMAIL_PASSWORD") # храни пароль в переменной окружения!

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        send_debug_message(f"[Email] Ошибка отправки: {e}")

def init_firebase():
    global db
    if db is None:
        try:
            print(" Инициализация Firebase...")
            firebase_config = os.getenv("FIREBASE_CONFIG")
            credentials_info = json.loads(firebase_config)
            cred = credentials.Certificate(credentials_info)
            initialize_app(cred)
            db = firestore.client()
            print(" Firebase успешно инициализирован")
        except Exception as e:
            print(f" Ошибка инициализации Firebase: {e}")

app = Flask(__name__)
CORS(app)

@app.before_request
def before_request():
    init_firebase()

# Названия моделей на Hugging Face
HF_MODELS = {
    "spam": "EgorU/rubert_spam_final",
    "toxic": "EgorU/rubert_toxic_model",
    "review": "EgorU/rubert_review_final",
    "sentiment": "EgorU/rubert_sentiment_model"
}

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def query_hf_model(model_key, text):
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODELS[model_key]}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 503:
        return "loading"
    else:
        print(f"Ошибка Hugging Face API ({model_key}): {response.text}")
        return None

ML_SERVER_URL = "https://ca72-91-103-252-33.ngrok-free.app/analyze"

def analyze_text(text):
    try:
        response = requests.post(ML_SERVER_URL, json={"text": text}, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print("Ошибка ML-сервера:", response.text)
            return {
                "is_spam": False,
                "spam_confidence": 0.0,
                "is_toxic": False,
                "toxic_confidence": 0.0,
                "is_review": False,
                "review_confidence": 0.0,
                "sentiment": None,
                "sentiment_confidence": 0.0
            }
    except Exception as e:
        print("Ошибка обращения к ML-серверу:", e)
        return {
            "is_spam": False,
            "spam_confidence": 0.0,
            "is_toxic": False,
            "toxic_confidence": 0.0,
            "is_review": False,
            "review_confidence": 0.0,
            "sentiment": None,
            "sentiment_confidence": 0.0
        }

@app.before_request
def before_request_log():
    print(f" NEW REQUEST: {request.method} {request.path}")
@app.route('/', methods=['POST', 'GET'])

def home():
    print("[DEBUG] Главная страница: что-то пришло!")
    print(request.data)
    return jsonify({"msg": "Hello from root"}), 200

# Обработка текста и выделение токсичных предложений
@app.route('/check', methods=['POST'])
def check_text():
    try:
        data = request.get_json()
        text = data.get('text')
        email = data.get('email')
        print(f"Полученный текст: {text}")
        print(f"Email пользователя: {email}")

        # Разбиваем текст на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        results = []
        for sent in sentences:
            if not sent.strip():
                continue
            result = analyze_text(sent)
            result_summary = {
                "text": sent,
                "is_safe": not (result["is_spam"] or result["is_toxic"]),
                "violations": [k for k in ["Спам", "Токсичность"] if (k == "Спам" and result["is_spam"]) or (k == "Токсичность" and result["is_toxic"])],
                "results": result,
                "is_review": result.get("is_review", False),
                "sentiment": result.get("sentiment", None)
            }
            results.append(result_summary)

        # Сохраняем одну запись на весь текст
        db.collection('checks').add({
            'text': text,
            'email': email,
            'sentences': results,
            'date': datetime.now()
        })

        return jsonify({"sentences": results})
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        # Получаем данные о проверках из Firestore
        checks = db.collection('checks').stream()
        data = []

        for check in checks:
            check_data = check.to_dict()
            data.append({
                'date': check_data['date'],  # Дата проверки
            })

        # Группируем проверки по дате
        stats = {}
        for item in data:
            date_str = item['date'].strftime('%Y-%m-%d')  # Форматируем дату
            stats[date_str] = stats.get(date_str, 0) + 1

        return jsonify({
            'stats': stats  # Возвращаем сгруппированные данные
        })
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check-url', methods=['POST'])
def check_url():
    data = request.json
    url = data.get('url')
    email = data.get('email')  # Получаем email из запроса
    if not url:
        return jsonify({'error': 'URL обязателен'}), 400

    try:
        # Добавляем заголовок User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)  # Используем заголовки
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')

        # Можно разбивать на предложения, если нужно, но здесь анализируем весь текст
        result = analyze_text(text)
        result_summary = {
            "url": url,
            "is_safe": not (result["is_spam"] or result["is_toxic"]),
            "violations": [k for k in ["spam", "toxic"] if result[f"is_{k}"]],
            "results": result
        }

        # Сохраняем результат анализа в Firestore
        db.collection('url_checks').add({
            'url': url,
            'email': email,  # Сохраняем email
            'result': result_summary,
            'date': datetime.now()
        })

        return jsonify(result_summary)
    except requests.RequestException as e:
        print(f"Ошибка при запросе URL: {e}")
        return jsonify({'error': f"Ошибка при запросе URL: {str(e)}"}), 500

# простая временная защита по message_id
recent_messages = set()

def detect_master(text):
    osokin_variants = [
        r"осокин(а|у|ом|е|ой)?", r"александр(а|у|ом|е|ом|ой)?", r"владимирович(а|у|ем|е)?",
        r"саша", r"саня", r"осокин а\.в\.", r"а\.в\. осокин"
    ]
    ivanov_variants = [
        r"иванов(а|у|ым|е|ой)?", r"виктор(а|у|ом|е|ом|ой)?", r"петрович(а|у|ем|е)?",
        r"виктор петрович", r"иванов в\.п\.", r"в\.п\. иванов"
    ]
    text_lower = text.lower()
    for pattern in osokin_variants:
        if re.search(pattern, text_lower):
            return "Осокин А.В."
    for pattern in ivanov_variants:
        if re.search(pattern, text_lower):
            return "Иванов В.П."
    return None

@app.route('/telegram-webhook', methods=['POST'])
def telegram_webhook():
    try:
        data = request.get_json()
        message = data.get('message')

        if not message:
            send_debug_message(" Нет message в payload!")
            return jsonify({"status": "no message"}), 200

        message_id = message.get('message_id')
        if message_id in recent_messages:
            send_debug_message(f" Дубликат message_id: {message_id}")
            return jsonify({"status": "duplicate"}), 200
        if message_id:
            recent_messages.add(message_id)

        # Информация о чате и пользователе
        chat = message['chat']
        group_id = str(chat['id'])
        group_title = chat.get('title', 'Без названия')
        from_user = message.get('from', {})
        user_id = from_user.get('id')
        author = f"{from_user.get('first_name', '')}_{from_user.get('last_name', '')}_{user_id}".strip("_")
        user_text = message.get('text', '')
        master = detect_master(user_text)

        # send_debug_message(f" Webhook получен от {author} в группе {group_title}\nТекст: {user_text}")

        if user_text.strip() == "/getid":
            telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
            telegram_api_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            requests.post(telegram_api_url, json={
                "chat_id": group_id,
                "text": f"ID группы: `{group_id}`\nНазвание: {group_title}",
                "parse_mode": "Markdown"
            })
            return jsonify({"status": "sent chat id"}), 200

        # Проверка регистрации группы
        group_doc = db.collection('groups').document(group_id).get()
        if not group_doc.exists:
            send_debug_message(f" Группа {group_title} не зарегистрирована.")
            return jsonify({"status": "group not registered"}), 200

        group_data = group_doc.to_dict() or {}
        admin_email = group_data.get('info', {}).get('admin_email')
        if not admin_email:
            send_debug_message(f" У группы {group_title} нет admin_email.")
            return jsonify({"status": "no admin email"}), 200

        # Новый подход: разбиваем на предложения и анализируем каждое
        sentences = re.split(r'(?<=[.!?])\s+', user_text.strip())
        results = []
        for sent in sentences:
            if not sent.strip():
                continue
            result = analyze_text(sent)
            result_summary = {
                "text": sent,
                "is_safe": not (result["is_spam"] or result["is_toxic"]),
                "violations": [k for k in ["Спам", "Токсичность"] if (k == "Спам" and result["is_spam"]) or (k == "Токсичность" and result["is_toxic"])],
                "results": result,
                "is_review": result.get("is_review", False),
                "sentiment": result.get("sentiment", None)
            }
            results.append(result_summary)

        # Вся проверка считается нарушающей, если хотя бы одно предложение с нарушением
        has_violation = any(sent["violations"] for sent in results)
        is_safe = not has_violation
        violations = []
        for sent in results:
            violations.extend(sent["violations"])
        violations = list(set(violations))
        # Для отзывов: если хотя бы одно предложение is_review
        review_flag = any(sent.get("is_review", False) for sent in results)
        # Для сентимента: если есть предложения с sentiment, можно взять по большинству или первому
        sentiment_flag = None
        sentiments = [sent.get("sentiment") for sent in results if sent.get("sentiment")]
        if sentiments:
            # Можно взять первый или посчитать по большинству
            sentiment_flag = max(set(sentiments), key=sentiments.count)

        # Сохраняем результат
        try:
            db.collection('groups').document(group_id).collection('checks').document().set({
                'text': user_text,
                'author': author,
                'review': review_flag,
                'sentiment': sentiment_flag,
                'sentences': results,
                'result': {
                    'is_safe': is_safe,
                    'violations': violations,
                    'results': results
                },
                'master': master,
                'date': datetime.now()
            })

            # Отправляем email только если есть токсичность или спам
            if ("Спам" in violations or "Токсичность" in violations):
                violations_text = ', '.join(violations) if violations else 'нет'
                email_body = (
                    f"Здравствуйте, в вашей Telegram-группе «{group_title}» ({group_id}) обнаружено подозрительное сообщение:\n\n"
                    f"Автор: {author}\n"
                    f"Текст: {user_text}\n"
                    f"Тип нарушения: {violations_text}\n"
                    f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"С уважением, ваша система мониторинга TextShield."
                )
                send_email(admin_email, " Обнаружено подозрительное сообщение", email_body)

        except Exception as e:
            send_debug_message(f" Ошибка при сохранении в Firestore: {e}")

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        error_msg = f" Ошибка в webhook: {str(e)}"
        send_debug_message(error_msg)
        return jsonify({"error": str(e)}), 500

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        file = request.files['file']
        upload_preset = 'diplom'
        
        # Формируем данные для отправки в Cloudinary
        form_data = {
            'file': file,
            'upload_preset': upload_preset
        }
        
        # Отправляем файл в Cloudinary
        response = requests.post(
            f'https://api.cloudinary.com/v1_1/dh2qb7atd/image/upload',
            files=form_data
        )
        
        # Получаем URL загруженного файла
        result = response.json()
        
        return jsonify({
            'url': result['secure_url']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f" Запуск сервера на порту: {port}")
    try:
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f" Ошибка запуска сервера: {e}")
