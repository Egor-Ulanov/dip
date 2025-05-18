from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import re
from functools import lru_cache
import time
import os
import smtplib
from email.mime.text import MIMEText
import json
import joblib
from tensorflow.keras.models import load_model

# Пути к файлам
REVIEW_MODEL_PATH = "review_detection_model.keras"
REVIEW_VECTORIZER_PATH = "review_vectorizer.pkl"
SENTIMENT_MODEL_PATH = "sentiment_model.keras"
SENTIMENT_VECTORIZER_PATH = "sentiment_vectorizer.pkl"

# Загрузка
review_model = load_model(REVIEW_MODEL_PATH)
review_vectorizer = joblib.load(REVIEW_VECTORIZER_PATH)

sentiment_model = load_model(SENTIMENT_MODEL_PATH)
sentiment_vectorizer = joblib.load(SENTIMENT_VECTORIZER_PATH)

# Инициализация Firebase
# Загрузка конфигурации из переменной окружения
firebase_config = os.getenv("FIREBASE_CONFIG")
credentials_info = json.loads(firebase_config)
cred = credentials.Certificate(credentials_info)

def is_review(text):
    try:
        X = review_vectorizer.transform([text])
        # send_debug_message(f"[ReviewCheck] Векторизованный текст: {X}")
        prediction = review_model.predict(X)[0][0]
        send_debug_message(f"[ReviewCheck] Предсказание: {prediction}")
        return prediction > 0.5
    except Exception as e:
        send_debug_message(f"[ReviewCheck] Ошибка определения отзыва: {e}")
        return False

def is_positive_review(text):
    try:
        X = sentiment_vectorizer.transform([text])
        prediction = sentiment_model.predict(X)[0][0]
        return prediction > 0.5
    except Exception as e:
        send_debug_message(f"[SentimentCheck] Ошибка определения тональности: {e}")
        return None

# print("Бот токен:",os.getenv("TELEGRAM_BOT_TOKEN"))

initialize_app(cred)
db = firestore.client()

# Токен API Hugging Face
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

app = Flask(__name__)
CORS(app)

# Функция для вызова Hugging Face Inference API
def query_huggingface_api(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    api_url = "https://api-inference.huggingface.co/models/SkolkovoInstitute/russian_toxicity_classifier"

    response = requests.post(api_url, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        try:
            result = response.json()
            print("Ответ Hugging Face API:", result)  # Отладочный вывод

            # Проверяем, является ли результат вложенным списком
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                result = result[0]  # Убираем вложенность

            # Убедимся, что результат — список словарей
            if isinstance(result, list) and all(isinstance(item, dict) for item in result):
                return result
            else:
                raise ValueError("Неверный формат данных от API")
        except ValueError as e:
            print(f"Ошибка: {str(e)}")
            return [{"label": "error", "score": 0.0}]  # Возвращаем заглушку для обработки ошибки
    elif response.status_code == 503:
        return [{"label": "loading", "score": 0.0}]  # Если модель загружается
    else:
        print(f"Ошибка API Hugging Face: {response.text}")
        return [{"label": "error", "score": 0.0}]  # Возвращаем заглушку

DEBUG_CHAT_ID = "-4661677635"  # ID твоего личного чата или тестовой группы
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
        time.sleep(0.3)  # 👈 не даём отправлять слишком быстро
    except Exception as e:
        print("Ошибка при отправке debug-сообщения:", e)

def send_email(to_email, subject, body):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = "egorulanov908@gmail.com"  # поменяй на свою почту
    password = os.getenv("EMAIL_PASSWORD")  # храни пароль в переменной окружения!

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
        email = data.get('email')  # Получаем email из запроса
        print(f"Полученный текст: {text}")
        print(f"Email пользователя: {email}")

        # Разбиваем текст на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        results = []
        violations = []

        is_safe = True  # По умолчанию текст безопасен

        for sentence in sentences:
            # Проверка каждого предложения через Hugging Face
            hf_result = query_huggingface_api(sentence)

            # Убедимся, что hf_result — это список словарей
            if not isinstance(hf_result, list) or not all(isinstance(pred, dict) for pred in hf_result):
                hf_result = [{"label": "error", "score": 0.0}]  # Заглушка в случае ошибки

            # Проверяем, является ли предложение токсичным
            is_toxic = any(pred["label"] == "toxic" and pred["score"] > 0.5 for pred in hf_result)
            if is_toxic:
                is_safe = False
                violations.append(sentence)

            results.append({
                "sentence": sentence,
                "is_toxic": is_toxic,
                "predictions": hf_result
            })

        result_summary = {
            "is_safe": is_safe,
            "violations": violations,
            "results": results
        }

        # Сохраняем в Firestore с добавлением email пользователя
        db.collection('checks').add({
            'text': text,
            'email': email,  # Сохраняем email
            'result': result_summary,
            'date': datetime.now()
        })

        return jsonify(result_summary)
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

        # Анализируем текст (вызываем Hugging Face API)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        results = []
        violations = []
        is_safe = True

        for sentence in sentences:
            hf_result = query_huggingface_api(sentence)

            # Убедимся, что hf_result корректен
            if not isinstance(hf_result, list) or not all(isinstance(pred, dict) for pred in hf_result):
                hf_result = [{"label": "error", "score": 0.0}]

            is_toxic = any(pred["label"] == "toxic" and pred["score"] > 0.5 for pred in hf_result)
            if is_toxic:
                is_safe = False
                violations.append(sentence)

            results.append({
                "sentence": sentence,
                "is_toxic": is_toxic,
                "predictions": hf_result
            })

        # Формируем результат анализа
        result_summary = {
            "url": url,
            "is_safe": is_safe,
            "violations": violations,
            "results": results
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

@app.route('/telegram-webhook', methods=['POST'])
def telegram_webhook():
    try:
        data = request.get_json()
        message = data.get('message')

        if not message:
            send_debug_message("❌ Нет message в payload!")
            return jsonify({"status": "no message"}), 200

        message_id = message.get('message_id')
        if message_id in recent_messages:
            send_debug_message(f"⚠️ Дубликат message_id: {message_id}")
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


        # send_debug_message(f"✅ Webhook получен от {author} в группе {group_title}\nТекст: {user_text}")

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
            send_debug_message(f"⚠️ Группа {group_title} не зарегистрирована.")
            return jsonify({"status": "group not registered"}), 200

        group_data = group_doc.to_dict() or {}
        admin_email = group_data.get('info', {}).get('admin_email')
        # send_debug_message(f"📦 group_data: {json.dumps(group_data, ensure_ascii=False)}")
        if not admin_email:
            send_debug_message(f"⚠️ У группы {group_title} нет admin_email.")
            return jsonify({"status": "no admin email"}), 200
        # Проверка токсичности
        sentences = re.split(r'(?<=[.!?])\s+', user_text)
        is_safe = True
        violations = []
        results = []

        for sentence in sentences:
            hf_result = query_huggingface_api(sentence)
            if not isinstance(hf_result, list):
                hf_result = [{"label": "error", "score": 0.0}]
            is_toxic = any(pred.get("label") == "toxic" and pred.get("score", 0) > 0.5 for pred in hf_result)
            if is_toxic:
                is_safe = False
                violations.append(sentence)
            results.append({
                "sentence": sentence,
                "is_toxic": is_toxic,
                "predictions": hf_result
            })

        # Проверка, является ли сообщение отзывом
        review_flag = is_review(user_text)
        sentiment_flag = None
        if review_flag:
            sentiment_flag = is_positive_review(user_text)
        
        # send_debug_message(f"📦 checks: {user_text, results}")
        # Сохраняем результат
        try:
            send_debug_message(f"📥 Сохраняю сообщение от {author}: {user_text}")
            db.collection('groups').document(group_id).collection('checks').document().set({
                'text': user_text,
                'author': author,
                'review': review_flag,
                'sentiment': sentiment_flag,
                'result': {
                    'is_safe': is_safe,
                    'violations': violations,
                    'results': results
                },
                'date': datetime.now()
            })
            send_debug_message(f"Сообщение review: {review_flag}, sentiment: {sentiment_flag}, is_safe: {is_safe}, violations: {violations},results: {results}")

            if not is_safe:
                email_body = (
                    f"В Telegram-группе «{group_title}» ({group_id}) "
                    f"обнаружено токсичное сообщение:\n\n"
                    f"Автор: {author}\n"
                    f"Текст: {user_text}\n\n"
                    f"Нарушения: {', '.join(violations)}"
                )
                send_email(admin_email, "⚠️ Обнаружено токсичное сообщение", email_body)
        except Exception as e:
            send_debug_message(f"❌ Ошибка при сохранении в Firestore: {e}")

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        error_msg = f"❌ Ошибка в webhook: {str(e)}"
        send_debug_message(error_msg)
        return jsonify({"error": str(e)}), 500

@app.route('/test-webhook', methods=['POST'])
def test_webhook():
    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({"status": "no message"}), 200

        chat = message['chat']
        user_text = message.get('text', '')
        chat_id = str(chat.get('id', 'unknown'))

        # Просто пишем в фиксированную коллекцию без проверок
        db.collection('groups').document("test").collection('checks').add({
            'text': user_text,
            'date': datetime.now()
        })

        return jsonify({"status": "saved"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send-test-email', methods=['GET'])
def send_test_email():
    test_email = request.args.get('to')
    if not test_email:
        return jsonify({"error": "Укажи ?to=example@mail.com в запросе"}), 400

    try:
        send_email(
            test_email,
            "Тестовое письмо от Flask",
            "Если ты это читаешь — отправка работает! ✅"
        )
        return jsonify({"status": f"Письмо отправлено на {test_email}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"✅ Starting server on port: {port}")
    app.run(host='0.0.0.0', port=port)
