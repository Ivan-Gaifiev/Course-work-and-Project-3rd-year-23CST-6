# reputation_analyzer_final.py
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from transformers import pipeline
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_sentiment_pipeline():
    print("Загрузка модели rubert-base-cased-sentiment (Blanchefort)...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
        tokenizer="blanchefort/rubert-base-cased-sentiment",
        device=-1,
        framework='pt'
    )
    print("Модель загружена.")
    return sentiment_pipeline

def analyze_sentiments(df, text_column='text', sentiment_pipeline=None):
    if sentiment_pipeline is None:
        sentiment_pipeline = load_sentiment_pipeline()
    
    print("Анализ тональности...")
    sentiments = []
    scores = []
    for text in tqdm(df[text_column], desc="Обработка"):
        if pd.isna(text) or str(text).strip() == "":
            sentiments.append("NEUTRAL")
            scores.append(0.0)
            continue
        try:
            truncated = text[:512]  # модель имеет ограничение
            result = sentiment_pipeline(truncated)[0]
            label = result['label'].lower()
            if label == "negative":
                sentiments.append("NEGATIVE")
            elif label == "positive":
                sentiments.append("POSITIVE")
            else:
                sentiments.append("NEUTRAL")
            scores.append(result['score'])
        except Exception as e:
            print(f"Ошибка: {text[:50]}... {e}")
            sentiments.append("NEUTRAL")
            scores.append(0.0)
    df['sentiment'] = sentiments
    df['confidence'] = scores
    return df

def download_dataset():
    # Встроенный демо-датасет (замените на реальный CSV)
    sample_data = {
        'text': [
            "Отличное приложение, всё работает быстро и удобно!",
            "Постоянно вылетает и тормозит, очень разочарован.",
            "Не очень нормальное приложение, нужно больше функций.",
            "Обнова всё сломала, теперь ничего не открывается.",
            "Лучшее приложение для общения!",
            "Рекламы слишком много, бесит.",
            "В целом неплохо, но баги есть.",
            "Спасибо разработчикам, всё супер!",
            "Не нравится новый дизайн, старый был лучше.",
            "Приложение достойное, пользуюсь каждый день."
        ],
        'rating': [5, 1, 3, 1, 5, 2, 3, 5, 2, 4]
    }
    df = pd.DataFrame(sample_data)
    print("Используется встроенный демо-датасет из 10 отзывов.")
    return df

def main():
    df = download_dataset()
    if 'text' not in df.columns:
        raise ValueError("Нет колонки 'text'")
    
    sentiment_pipeline = load_sentiment_pipeline()
    df = analyze_sentiments(df, 'text', sentiment_pipeline)
    
    output_file = "reviews_with_sentiment.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nРезультат сохранён в {output_file}")
    
    print("\n=== Статистика тональности ===")
    print(df['sentiment'].value_counts())
    
    for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        examples = df[df['sentiment'] == sentiment]
        if not examples.empty:
            print(f"\nПримеры {sentiment} отзывов (первые 2):")
            for _, row in examples.head(2).iterrows():
                print(f"  - {row['text'][:100]}... (уверенность: {row['confidence']:.2f})")

if __name__ == "__main__":
    main()