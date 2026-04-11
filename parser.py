import pandas as pd
import sqlite3
from google_play_scraper import Sort, reviews


def fetch_and_store_data():
    print("Начинаю сбор данных...")
    sort=Sort.NEWEST,  # Берем самые свежие отзывы

    # 1. СБОР ДАННЫХ (Extract)
    # Вместо 'com.vkontakte.android' будет приложение вашей целевой компании
    result, continuation_token = reviews(
        'com.vkontakte.android',
        lang='ru',
        country='ru',
        count=200  # Количество отзывов для тестовой выборки
    )

    # 2. ОБРАБОТКА ДАННЫХ (Transform)
    # Преобразуем JSON-ответ в удобный DataFrame
    df = pd.DataFrame(result)

    # Оставляем только нужные для аналитики колонки
    df = df[['at', 'score', 'content', 'userName']]
    df.rename(columns={
        'at': 'date',
        'score': 'rating',
        'content': 'text',
        'userName': 'author'
    }, inplace=True)

    # Убираем пустые строки, если они есть
    df.dropna(subset=['text'], inplace=True)

    print(f"Успешно собрано и обработано {len(df)} отзывов.")

    # 3. ХРАНЕНИЕ ДАННЫХ (Load)
    # Выгрузка в CSV
    csv_filename = 'company_reviews_raw.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"Данные сохранены в файл: {csv_filename} (Готово для передачи ML-команде)")

    # Сохранение в базу данных SQLite для визуализации
    db_filename = 'reputation_monitoring.db'
    conn = sqlite3.connect(db_filename)
    # Сохраняем DataFrame в таблицу 'mentions'
    df.to_sql('mentions', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Данные загружены в базу данных: {db_filename} (Готово для BI/Дашбордов)")

    return df


# Запуск прототипа
if __name__ == "__main__":
    sample_df = fetch_and_store_data()

    # Показываем первые 3 отзыва для демонстрации
    print("\nПример собранных данных:")
    print(sample_df.head(3))