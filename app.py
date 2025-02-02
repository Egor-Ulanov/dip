from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import re
import os
import json

# Инициализация Firebase
# Загрузка конфигурации из переменной окружения
firebase_config = os.getenv("FIREBASE_CONFIG")
credentials_info = json.loads(firebase_config)
cred = credentials.Certificate(credentials_info)

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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
