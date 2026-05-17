import pandas as pd
from sqlalchemy import create_engine, text
from google_play_scraper import search, reviews
import requests
import datetime

# --- НАСТРОЙКИ ---
DB_URL = "postgresql://postgres.lfqejjtoeszbjrihfhfv:ktWwZiIPevuP6H60@aws-1-eu-north-1.pooler.supabase.com:6543/postgres"
engine = create_engine(DB_URL)

# ==================== ФУНКЦИИ РАБОТЫ С БД ====================
def get_or_create_company(legal_name):
    with engine.begin() as conn:
        res = conn.execute(text("SELECT company_id FROM companies WHERE legal_name = :name"), {"name": legal_name}).fetchone()
        if res: return res[0]
        res = conn.execute(text("INSERT INTO companies (legal_name) VALUES (:name) RETURNING company_id"), {"name": legal_name}).fetchone()
        return res[0]

def get_or_create_source(source_name):
    with engine.begin() as conn:
        res = conn.execute(text("SELECT source_id FROM sources WHERE name = :name"), {"name": source_name}).fetchone()
        if res: return res[0]
        res = conn.execute(text("INSERT INTO sources (name, type) VALUES (:name, 'review') RETURNING source_id"), {"name": source_name}).fetchone()
        return res[0]

def create_search_task(company_id, source_id):
    with engine.begin() as conn:
        res = conn.execute(text("""
            INSERT INTO search_tasks (company_id, source_id, status) 
            VALUES (:cid, :sid, 'running') RETURNING task_id
        """), {"cid": company_id, "sid": source_id}).fetchone()
        return res[0]

# ==================== ФУНКЦИИ ПАРСИНГА ====================
def get_ids(company_name):
    gp_id, as_id = None, None
    
    # Создаем специальный запрос для Google Play
    gp_search_query = company_name
    if not gp_search_query.startswith("id="):
        gp_search_query = f"id={gp_search_query}"
    
    # Для Apple наоборот — убираем "id=", если вдруг пользователь его ввел
    apple_search_query = company_name.replace("id=", "")

    try:
        # Используем модифицированный запрос для Google
        gp_res = search(gp_search_query, lang="ru", country="ru")
        if gp_res:
            gp_id = gp_res[0]['appId']
    except Exception as e:
        print(f"Ошибка поиска GP: {e}")
    
    try:
        # Используем чистый запрос для Apple
        url = f"https://itunes.apple.com/search?term={apple_search_query}&country=ru&entity=software"
        apple_res = requests.get(url, timeout=5).json()
        if apple_res.get('results'):
            as_id = apple_res['results'][0]['trackId']
    except Exception as e:
        print(f"Ошибка поиска AppStore: {e}")
    
    return gp_id, as_id

def scrape_google_play(app_id, count=100):
    if not app_id: return pd.DataFrame()
    result, _ = reviews(app_id, lang='ru', country='ru', count=count)
    df = pd.DataFrame(result)
    if df.empty: return df
    df = df[['at', 'score', 'content', 'userName']].rename(columns={'at': 'date', 'score': 'rating', 'content': 'text', 'userName': 'author'})
    df['url'], df['title'] = '', ''
    return df

def scrape_app_store_rss_bulk(app_id, count=100):
    if not app_id: return pd.DataFrame()
    all_reviews = []
    headers = {"User-Agent": "Mozilla/5.0"}
    max_pages = min(10, (count // 50) + 1)

    for page in range(1, max_pages + 1):
        url = f"https://itunes.apple.com/ru/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: break
            entries = response.json().get('feed', {}).get('entry', [])
            if not entries: break
            if isinstance(entries, dict): entries = [entries]

            for entry in entries:
                if 'im:rating' not in entry: continue
                all_reviews.append({
                    'date': entry.get('updated', {}).get('label'),
                    'rating': int(entry.get('im:rating', {}).get('label', 0)),
                    'text': entry.get('content', {}).get('label', ''),
                    'author': entry.get('author', {}).get('name', {}).get('label', 'Аноним'),
                    'title': entry.get('title', {}).get('label', ''),
                    'url': ''
                })
        except: break
    return pd.DataFrame(all_reviews)

def save_mentions(df, company_id, source_id):
    """Сохранение собранных данных в базу"""
    if df is None or df.empty: return
    task_id = create_search_task(company_id, source_id)
    df['task_id'] = task_id
    df['company_id'] = company_id
    df['source_id'] = source_id
    df['sentiment_score'] = None
    df['parsed_at'] = datetime.datetime.now(datetime.timezone.utc)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    columns_to_save = ['task_id', 'company_id', 'source_id', 'url', 'title', 'author', 'date', 'rating', 'text', 'sentiment_score', 'parsed_at']
    df[columns_to_save].to_sql('mentions', engine, if_exists='append', index=False)
    
    with engine.begin() as conn:
        conn.execute(text("UPDATE search_tasks SET status = 'completed', total_mentions = :count WHERE task_id = :tid"), 
                     {"count": len(df), "tid": task_id})