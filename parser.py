import pandas as pd
from sqlalchemy import create_engine, text
from google_play_scraper import search, reviews
import requests
import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time


# --- НАСТРОЙКИ ---
DB_URL = "postgresql://postgres.lfqejjtoeszbjrihfhfv:ktWwZiIPevuP6H60@aws-1-eu-north-1.pooler.supabase.com:6543/postgres"
engine = create_engine(DB_URL)

VK_TOKEN = "d5d65e18d5d65e18d5d65e180bd6974bccdd5d6d5d65e18bfda21af129793221ee595c0"
VK_API_V = "5.131"

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

def scrape_yandex_maps(company_name, count=50):
    """Сбор отзывов с Яндекс.Карт через Selenium (без API)"""
    if not company_name:
        return pd.DataFrame()

    print(f"🔍 Яндекс.Карты: ищу организацию '{company_name}' через браузер...")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    all_reviews = []
    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # 1. Переходим на Яндекс.Карты и ищем организацию
        search_url = f"https://yandex.ru/maps/?text={company_name}"
        driver.get(search_url)
        time.sleep(3)
        
        # 2. Кликаем на первую найденную организацию
        try:
            # Ищем первую карточку организации в результатах поиска
            first_result = driver.find_element(By.CSS_SELECTOR, '.search-list-view__item')
            first_result.click()
            time.sleep(2)
        except:
            print("❌ Организация не найдена в результатах поиска")
            driver.quit()
            return pd.DataFrame()
        
        # 3. Переходим во вкладку "Отзывы"
        try:
            reviews_tab = driver.find_element(By.CSS_SELECTOR, '[href$="/reviews/"]')
            reviews_tab.click()
            time.sleep(3)
        except:
            print("❌ Вкладка с отзывами не найдена")
            driver.quit()
            return pd.DataFrame()
        
        # 4. Собираем отзывы с прокруткой
        print("⏳ Собираю отзывы...")
        last_height = -1
        while len(all_reviews) < count:
            # Прокручиваем контейнер с отзывами
            driver.execute_script(
                "var scroll = document.querySelector('.scroll__container');"
                "if(scroll) scroll.scrollTop = scroll.scrollHeight;"
            )
            time.sleep(1.5)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            review_blocks = soup.find_all('div', class_='business-review-view__info')
            
            if not review_blocks:
                print("⚠️ Блоки отзывов не найдены")
                break
            
            for block in review_blocks:
                if len(all_reviews) >= count:
                    break
                    
                # Текст отзыва
                text_tag = block.find('span', class_='business-review-view__body-text')
                if not text_tag:
                    continue
                text = text_tag.text.strip()
                
                # Оценка (звёзды)
                rating_div = block.find('div', class_='business-rating-badge-view__stars')
                rating = 0
                if rating_div:
                    full_stars = rating_div.find_all('span', class_=lambda x: x and '_full' in x)
                    rating = len(full_stars)
                
                # Дата
                date_tag = block.find('span', class_='business-review-view__date')
                date_str = date_tag.text if date_tag else None
                
                # Автор
                author_tag = block.find('div', class_='business-review-view__author')
                author = author_tag.find('span').text if author_tag and author_tag.find('span') else 'Аноним'
                
                all_reviews.append({
                    'date': pd.to_datetime(date_str, errors='coerce') if date_str else datetime.datetime.now(),
                    'rating': rating,
                    'text': text,
                    'author': author,
                    'title': '',
                    'url': driver.current_url
                })
            
            # Проверяем, изменилась ли высота
            new_height = driver.execute_script(
                "var scroll = document.querySelector('.scroll__container');"
                "return scroll ? scroll.scrollHeight : 0;"
            )
            if new_height == last_height:
                print("ℹ️ Больше отзывов не грузится")
                break
            last_height = new_height
        
        driver.quit()
        print(f"✅ Собрано {len(all_reviews)} отзывов с Яндекс.Карт")
        
    except Exception as e:
        print(f"❌ Ошибка Selenium Яндекс.Карт: {e}")
        if driver:
            driver.quit()
        return pd.DataFrame()
    
    return pd.DataFrame(all_reviews)

def scrape_vk_newsfeed(company_name, count=50):
    """Глобальный поиск упоминаний компании по всему ВКонтакте (стены, новости, комментарии)"""
    if not company_name: return pd.DataFrame()
    
    url = "https://api.vk.com/method/newsfeed.search"
    params = {
        "q": company_name,
        "count": min(count, 200), # VK отдает максимум 200 за раз
        "access_token": VK_TOKEN,
        "v": VK_API_V
    }
    
    all_mentions = []
    try:
        res = requests.get(url, params=params, timeout=10).json()
        items = res.get('response', {}).get('items', [])
        
        for item in items:
            if not item.get('text'): continue
            
            owner_id = item.get('owner_id')
            post_id = item.get('id')
            
            all_mentions.append({
                'date': datetime.datetime.fromtimestamp(item['date']),
                'rating': None,
                'text': item['text'],
                'author': f"ID владельца: {owner_id}",
                'title': 'Глобальное упоминание в VK',
                'url': f"https://vk.com/wall{owner_id}_{post_id}"
            })
    except Exception as e:
        print(f"Ошибка VK Newsfeed: {e}")
        
    return pd.DataFrame(all_mentions)


def scrape_habr_rss(company_name, count=50):
    """Парсинг упоминаний компании из официальной поисковой RSS-ленты Хабра с поддержкой пагинации"""
    if not company_name: return pd.DataFrame()
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ReputationParser/1.0"}
    all_mentions = []
    page = 1

    while len(all_mentions) < count:
        # Добавляем параметр &page= для обхода ограничения в 20 статей
        url = f"https://habr.com/ru/rss/search/?q={company_name}&target_type=posts&order_by=date&page={page}"
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                break
                
            soup = BeautifulSoup(res.text, 'xml') 
            items = soup.find_all('item')
            
            # Если на странице нет статей (дошли до конца выдачи), останавливаем цикл
            if not items:
                break
                
            for item in items:
                if len(all_mentions) >= count:
                    break
                    
                title = item.find('title').text if item.find('title') else 'Статья на Хабре'
                link = item.find('link').text if item.find('link') else ''
                desc = item.find('description').text if item.find('description') else ''
                pub_date = item.find('pubDate').text if item.find('pubDate') else None
                
                clean_text = BeautifulSoup(desc, "html.parser").get_text(separator=' ') if desc else title
                
                all_mentions.append({
                    'date': pd.to_datetime(pub_date, errors='coerce') if pub_date else datetime.datetime.now(),
                    'rating': None,
                    'text': clean_text,
                    'author': 'Автор Хабра',
                    'title': title,
                    'url': link
                })
                
            # Переходим на следующую страницу (каждая страница даст еще +20 статей)
            page += 1
            
        except Exception as e:
            print(f"Ошибка Хабра на странице {page}: {e}")
            break
            
    return pd.DataFrame(all_mentions)


def scrape_google_news(company_name, count=50):
    """Сбор новостей через Google News RSS (с фильтрацией по ключевым словам)"""
    if not company_name or not company_name.strip():
        return pd.DataFrame()
    
    query_encoded = requests.utils.quote(company_name)
    url = f"https://news.google.com/rss/search?q={query_encoded}&hl=ru&gl=RU&ceid=RU:ru"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    all_news = []
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return pd.DataFrame()
        soup = BeautifulSoup(res.text, 'xml')
        items = soup.find_all('item')
        query_words = company_name.lower().split()
        
        for item in items:
            if len(all_news) >= count:
                break
            title = item.find('title').text if item.find('title') else ''
            link = item.find('link').text if item.find('link') else ''
            pub_date = item.find('pubDate').text if item.find('pubDate') else None
            source_tag = item.find('source')
            source = source_tag.text if source_tag else ''
            desc = item.find('description').text if item.find('description') else ''
            clean_text = BeautifulSoup(desc, 'html.parser').get_text(separator=' ')
            # Проверка релевантности: запрос в заголовке или тексте
            title_lower, text_lower = title.lower(), clean_text.lower()
            if any(word in title_lower or word in text_lower for word in query_words):
                all_news.append({
                    'date': pd.to_datetime(pub_date, errors='coerce') if pub_date else datetime.datetime.now(),
                    'rating': None,
                    'text': clean_text[:500],
                    'author': source,
                    'title': title,
                    'url': link
                })
    except Exception as e:
        print(f"Ошибка Google News RSS: {e}")
    return pd.DataFrame(all_news)

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
                # ВОТ ЭТА ПРОВЕРКА: останавливаемся, как только набрали нужное количество
                if len(all_reviews) >= count:
                    break
                    
                if 'im:rating' not in entry: continue
                all_reviews.append({
                    'date': entry.get('updated', {}).get('label'),
                    'rating': int(entry.get('im:rating', {}).get('label', 0)),
                    'text': entry.get('content', {}).get('label', ''),
                    'author': entry.get('author', {}).get('name', {}).get('label', 'Аноним'),
                    'title': entry.get('title', {}).get('label', ''),
                    'url': ''
                })
                
            # Если уже набрали лимит во внутреннем цикле, выходим и из внешнего
            if len(all_reviews) >= count:
                break
                
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