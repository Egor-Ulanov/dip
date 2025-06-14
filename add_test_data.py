import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import json
import os

# Получаем путь к файлу конфигурации Firebase из переменной окружения
firebase_config = {
    "type": "service_account",
    "project_id": "dipl-12202",
    "private_key_id": "3030a99552",
    "client_email": "firebase-adminsdk-dn0om@dipl-12202.iam.gserviceaccount.com",
    "token_uri": "https://oauth2.googleapis.com/token",
    "universe_domain": "googleapis.com"
}

# Инициализация Firebase
cred = credentials.Certificate('dipl-12202-firebase-adminsdk-dn0om-3030a99552.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Тестовые данные
test_data = [

    {
        "text": "в какие дни вы работаете?",
        "author": "Егор Уланов__5016903200",
        "master": None,
        "result": {
            "is_safe": True,
            "violations": [],
            "results": []
        },
        "sentiment": None,
        "spam": True,
        "date": datetime(2025, 6, 8)
    }
]

def add_test_data():
    group_id = "-4916466343"  # ID тестовой группы
    
    # Добавляем каждую запись
    for data in test_data:
        try:
            # Создаем новый документ в коллекции checks
            doc_ref = db.collection('groups').document(group_id).collection('checks').document()
            doc_ref.set(data)
            print(f'Добавлена запись для мастера: {data["master"]}')
        except Exception as e:
            print(f'Ошибка при добавлении записи: {e}')

if __name__ == '__main__':
    add_test_data()
    print('Все тестовые данные добавлены') 