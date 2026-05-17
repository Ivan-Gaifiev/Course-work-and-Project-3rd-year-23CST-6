import os
import pandas as pd
from transformers import pipeline
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from datetime import datetime
import warnings
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"

DB_URL = "postgresql://postgres.lfqejjtoeszbjrihfhfv:ktWwZiIPevuP6H60@aws-1-eu-north-1.pooler.supabase.com:5432/postgres?connect_timeout=30"

engine = create_engine(
    DB_URL,
    poolclass=NullPool,
    pool_pre_ping=True,
    connect_args={
        "options": "-c prepared_statement_cache_size=0",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 5,
        "keepalives_count": 5
    }
)

def analyze_reviews_incrementally(batch_size=200):
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="sunny3/rubert-conversational-sentiment-balanced",
        device=-1,
        framework='pt'
    )

    # 1. Подсчёт статистики
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(mention_id) FROM mentions WHERE text IS NOT NULL")).scalar()
        already = conn.execute(text("SELECT COUNT(DISTINCT mention_id) FROM sentiments")).scalar()

    remaining = total - already
    if remaining == 0:
        print("Все отзывы уже проанализированы!")
        return

    offset = 0
    processed = 0
    start_time = time.time()

    while processed < remaining:
        query = text("""
            SELECT m.mention_id, m.text, m.author, m.rating
            FROM mentions m
            LEFT JOIN sentiments s ON m.mention_id = s.mention_id
            WHERE m.text IS NOT NULL AND s.mention_id IS NULL
            LIMIT :batch_size OFFSET :offset
        """)
        df = pd.read_sql(query, engine, params={"batch_size": batch_size, "offset": offset})
        if df.empty:
            break

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Анализ отзывов"):
            text_val = row['text'][:512] if len(row['text']) > 512 else row['text']
            try:
                result = sentiment_pipeline(text_val)[0]
                label = result['label'].lower()
                confidence = result['score']

                if label == "positive":
                    sentiment = "POSITIVE"
                    score = 1.0
                elif label == "negative":
                    sentiment = "NEGATIVE"
                    score = -1.0
                else:
                    sentiment = "NEUTRAL"
                    score = 0.0

                results.append({
                    "mention_id": row['mention_id'],
                    "score": score,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "model_version": "rubert-conversational-sentiment-balanced",
                    "analyzed_at": datetime.utcnow()
                })
            except Exception as e:
                continue

        if results:
            inserted = 0
            while inserted == 0:
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            text("""
                                INSERT INTO sentiments 
                                (mention_id, score, sentiment, confidence, model_version, analyzed_at)
                                VALUES (:mention_id, :score, :sentiment, :confidence, :model_version, :analyzed_at)
                            """),
                            results
                        )
                    inserted = len(results)
                except Exception as e:
                    time.sleep(5)
            processed += inserted

        offset += batch_size
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0

if __name__ == "__main__":
    analyze_reviews_incrementally(batch_size=200)
