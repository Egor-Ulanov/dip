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

def init_firebase():
    global db
    if db is None:
        try:
            print("🔄 Инициализация Firebase...")
            firebase_config = os.getenv("FIREBASE_CONFIG")
            credentials_info = json.loads(firebase_config)
            cred = credentials.Certificate(credentials_info)
            initialize_app(cred)
            db = firestore.client()
            print("✅ Firebase успешно инициализирован")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации Firebase: {e}")

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

def analyze_text(text):
    # Проверка на спам
    spam_result = query_hf_model("spam", text)
    is_spam = False
    spam_conf = 0.0
    if isinstance(spam_result, list):
        for pred in spam_result:
            if pred.get("label") in ["spam", "LABEL_1"]:
                is_spam = pred.get("score", 0) > 0.5
                spam_conf = pred.get("score", 0)
                break

    # Проверка на токсичность
    toxic_result = query_hf_model("toxic", text)
    is_toxic = False
    toxic_conf = 0.0
    if isinstance(toxic_result, list):
        for pred in toxic_result:
            if pred.get("label") in ["toxic", "LABEL_1"]:
                is_toxic = pred.get("score", 0) > 0.5
                toxic_conf = pred.get("score", 0)
                break

    # Проверка, является ли текст отзывом
    review_result = query_hf_model("review", text)
    is_review = False
    review_conf = 0.0
    if isinstance(review_result, list):
        for pred in review_result:
            if pred.get("label") in ["LABEL_1", "review"]:
                is_review = pred.get("score", 0) > 0.5
                review_conf = pred.get("score", 0)
                break

    # Если это отзыв — проверяем сентимент
    sentiment = None
    sentiment_conf = 0.0
    if is_review:
        sentiment_result = query_hf_model("sentiment", text)
        if isinstance(sentiment_result, list):
            for pred in sentiment_result:
                if pred.get("label") in ["LABEL_1", "positive"]:
                    sentiment = "positive"
                    sentiment_conf = pred.get("score", 0)
                elif pred.get("label") in ["LABEL_0", "negative"]:
                    sentiment = "negative"
                    sentiment_conf = pred.get("score", 0)

    return {
        "is_spam": is_spam,
        "spam_confidence": spam_conf,
        "is_toxic": is_toxic,
        "toxic_confidence": toxic_conf,
        "is_review": is_review,
        "review_confidence": review_conf,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_conf
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

        # Можно разбивать на предложения, если нужно, но здесь анализируем весь текст
        result = analyze_text(text)
        result_summary = {
            "is_safe": not (result["is_spam"] or result["is_toxic"]),
            "violations": [k for k in ["spam", "toxic"] if result[f"is_{k}"]],
            "results": result
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
        if not admin_email:
            send_debug_message(f"⚠️ У группы {group_title} нет admin_email.")
            return jsonify({"status": "no admin email"}), 200

        # Проверка на спам
        spam_check = analyze_text(user_text)
        
        # Проверка токсичности
        sentences = re.split(r'(?<=[.!?])\s+', user_text)
        is_safe = True
        violations = []
        results = []

        for sentence in sentences:
            hf_result = query_hf_model("toxic", sentence)
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
        review_flag = analyze_text(user_text)["is_review"]
        if review_flag:
            sentiment_flag = analyze_text(user_text)["sentiment"]
        
        # send_debug_message(f"📦 checks: {user_text, results}")
        # Сохраняем результат
        try:
            db.collection('groups').document(group_id).collection('checks').document().set({
                'text': user_text,
                'author': author,
                'review': review_flag,
                'sentiment': sentiment_flag,
                'spam_check': spam_check,  # Добавляем результат проверки на спам
                'result': {
                    'is_safe': is_safe,
                    'violations': violations,
                    'results': results
                },
                'date': datetime.now()
            })

            if not is_safe or spam_check['is_spam']:
                violations_text = ', '.join(violations) if violations else 'нет'
                email_body = (
                    f"В Telegram-группе «{group_title}» ({group_id}) "
                    f"обнаружено проблемное сообщение:\n\n"
                    f"Автор: {author}\n"
                    f"Текст: {user_text}\n\n"
                    f"Токсичность: {'Обнаружена' if not is_safe else 'Не обнаружена'}\n"
                    f"Спам: {'Обнаружен' if spam_check['is_spam'] else 'Не обнаружен'}\n"
                    f"Уверенность (спам): {spam_check['spam_confidence']:.2%}\n"
                    f"Нарушения: {violations_text}"
                )
                send_email(admin_email, "⚠️ Обнаружено проблемное сообщение", email_body)

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
    print(f"✅ Запуск сервера на порту: {port}")
    try:
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
