import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from google_play_scraper import Sort, reviews
from transformers import pipeline
from tqdm import tqdm
import time
import datetime
from sqlalchemy import text
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter

from parser import get_ids, get_or_create_company, get_or_create_source, scrape_google_play, scrape_app_store_rss_bulk, save_mentions
from reputation_monitor import analyze_reviews_incrementally, engine
from topic_analyzer import get_word_frequencies_simple, get_word_frequencies_lemmatized, get_best_topics, run_all_models_comparison, lda_topic_model, nmf_topic_model, bertopic_model


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# ФУНКЦИЯ ДЛЯ ПРОВЕРКИ СУЩЕСТВУЮЩИХ ОТЗЫВОВ
def check_existing_reviews(df, company_id, source_id):
    """Проверяет, какие отзывы уже есть в БД"""
    if df.empty:
        return df
    
    with engine.connect() as conn:
        # Получаем все существующие отзывы для этой компании и источника
        query = text("""
            SELECT DISTINCT text, author, date 
            FROM mentions 
            WHERE company_id = :company_id AND source_id = :source_id
        """)
        existing = conn.execute(query, {"company_id": company_id, "source_id": source_id}).fetchall()
        
        # Создаем множество существующих комбинаций
        existing_set = set()
        for row in existing:
            text_normalized = ' '.join(row[0].strip().split()) if row[0] else ''
            author_normalized = row[1].strip() if row[1] else ''
            date_normalized = row[2] if row[2] else None
            existing_set.add((text_normalized, author_normalized, date_normalized))
    
    # Фильтруем новые отзывы
    new_rows = []
    for _, row in df.iterrows():
        text_normalized = ' '.join(str(row['text']).strip().split()) if pd.notna(row['text']) else ''
        author_normalized = str(row['author']).strip() if pd.notna(row['author']) else ''
        date_normalized = row['date'] if pd.notna(row['date']) else None
        
        if (text_normalized, author_normalized, date_normalized) not in existing_set:
            new_rows.append(row)
    
    if new_rows:
        return pd.DataFrame(new_rows)
    return pd.DataFrame()

# МОДИФИЦИРОВАННАЯ ФУНКЦИЯ get_recent_reviews С КЕШИРОВАНИЕМ
@st.cache_data(ttl=300, show_spinner=False)
def get_recent_reviews_cached(company_id, limit=1000):
    """Кешированная версия получения отзывов"""
    with engine.connect() as conn:
        query = text("""
            SELECT DISTINCT ON (m.text, m.author, m.date)
                m.text,
                s.sentiment,
                m.author,
                m.date,
                m.rating,
                src.name as source,
                m.mention_id
            FROM mentions m
            JOIN sentiments s ON m.mention_id = s.mention_id
            JOIN sources src ON m.source_id = src.source_id
            WHERE m.text IS NOT NULL 
                AND s.sentiment IS NOT NULL
                AND m.company_id = :company_id
            ORDER BY m.text, m.author, m.date, m.date DESC
            LIMIT :limit
        """)
        result = conn.execute(query, {"company_id": company_id, "limit": limit})
        rows = result.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['text', 'sentiment', 'author', 'date', 'rating', 'source', 'mention_id'])
            df = df.drop_duplicates(subset=['text', 'author'], keep='first')
            return df
    return pd.DataFrame()

# функции для работы с БД
def get_sentiment_stats(company_id=None):
    """Получение статистики по тональности отзывов (уникальных)"""
    with engine.connect() as conn:
        if company_id:
            query = text("""
                SELECT DISTINCT ON (m.text, m.author)
                    s.sentiment
                FROM sentiments s
                JOIN mentions m ON s.mention_id = m.mention_id
                WHERE m.company_id = :company_id
            """)
            result = conn.execute(query, {"company_id": company_id})
            sentiments = [row[0] for row in result.fetchall()]
            
            return {
                'positive_count': sentiments.count('POSITIVE'),
                'neutral_count': sentiments.count('NEUTRAL'),
                'negative_count': sentiments.count('NEGATIVE'),
                'total_analyzed': len(sentiments)
            }
        else:
            query = text("""
                SELECT DISTINCT ON (m.text, m.author)
                    s.sentiment
                FROM sentiments s
                JOIN mentions m ON s.mention_id = m.mention_id
            """)
            result = conn.execute(query)
            sentiments = [row[0] for row in result.fetchall()]
            
            return {
                'positive_count': sentiments.count('POSITIVE'),
                'neutral_count': sentiments.count('NEUTRAL'),
                'negative_count': sentiments.count('NEGATIVE'),
                'total_analyzed': len(sentiments)
            }
    return None

def get_daily_trends(company_id=None, days=30):
    """Получение динамики тональности по дням"""
    with engine.connect() as conn:
        if company_id:
            query = text("""
                SELECT 
                    DATE(m.date) as review_date,
                    COUNT(DISTINCT CASE WHEN s.sentiment = 'POSITIVE' THEN m.text || '||' || m.author END) as positive_count,
                    COUNT(DISTINCT CASE WHEN s.sentiment = 'NEUTRAL' THEN m.text || '||' || m.author END) as neutral_count,
                    COUNT(DISTINCT CASE WHEN s.sentiment = 'NEGATIVE' THEN m.text || '||' || m.author END) as negative_count,
                    COUNT(DISTINCT m.text || '||' || m.author) as total_reviews
                FROM sentiments s
                JOIN mentions m ON s.mention_id = m.mention_id
                WHERE m.date >= CURRENT_DATE - INTERVAL ':days days'
                    AND m.company_id = :company_id
                GROUP BY DATE(m.date) 
                ORDER BY review_date
            """)
            result = conn.execute(query, {"days": days, "company_id": company_id})
        else:
            query = text("""
                SELECT 
                    DATE(m.date) as review_date,
                    COUNT(DISTINCT CASE WHEN s.sentiment = 'POSITIVE' THEN m.text || '||' || m.author END) as positive_count,
                    COUNT(DISTINCT CASE WHEN s.sentiment = 'NEUTRAL' THEN m.text || '||' || m.author END) as neutral_count,
                    COUNT(DISTINCT CASE WHEN s.sentiment = 'NEGATIVE' THEN m.text || '||' || m.author END) as negative_count,
                    COUNT(DISTINCT m.text || '||' || m.author) as total_reviews
                FROM sentiments s
                JOIN mentions m ON s.mention_id = s.mention_id
                WHERE m.date >= CURRENT_DATE - INTERVAL ':days days'
                GROUP BY DATE(m.date) 
                ORDER BY review_date
            """)
            result = conn.execute(query, {"days": days})
        
        rows = result.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['review_date', 'positive_count', 'neutral_count', 'negative_count', 'total_reviews'])
            df['positive_pct'] = (df['positive_count'] / df['total_reviews'] * 100).fillna(0)
            df['negative_pct'] = (df['negative_count'] / df['total_reviews'] * 100).fillna(0)
            df['neutral_pct'] = (df['neutral_count'] / df['total_reviews'] * 100).fillna(0)
            return df
    
    return pd.DataFrame()

def get_recent_reviews(company_id=None, limit=10):
    """Получение последних отзывов с анализом"""
    if company_id:
        return get_recent_reviews_cached(company_id, limit)
    return pd.DataFrame()

def get_word_frequencies(company_id=None, limit=500):
    """Получение частотности слов"""
    with engine.connect() as conn:
        if company_id:
            query = text("""
                SELECT DISTINCT ON (m.text, m.author) m.text
                FROM mentions m
                JOIN sentiments s ON m.mention_id = s.mention_id
                WHERE m.text IS NOT NULL
                    AND m.company_id = :company_id
                LIMIT :limit
            """)
            result = conn.execute(query, {"company_id": company_id, "limit": limit})
        else:
            query = text("""
                SELECT DISTINCT ON (m.text, m.author) m.text
                FROM mentions m
                JOIN sentiments s ON m.mention_id = s.mention_id
                WHERE m.text IS NOT NULL
                LIMIT :limit
            """)
            result = conn.execute(query, {"limit": limit})
        
        rows = result.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['text'])
            
            try:
                russian_stop_words = set(stopwords.words('russian'))
            except:
                nltk.download('stopwords')
                russian_stop_words = set(stopwords.words('russian'))
            
            extra_stop_words = {
                'это', 'этот', 'эта', 'эти', 'этого', 'этому', 'этим', 'этом',
                'весь', 'вся', 'все', 'всё', 'всего', 'всем', 'всеми', 'всех',
                'такой', 'такая', 'такое', 'такие', 'такого', 'такой', 'таких'
            }
            
            stop_words = russian_stop_words.union(extra_stop_words)
            
            all_text = ' '.join(df['text'].tolist())
            all_text = re.sub(r'[^\w\sа-яА-Я]', '', all_text.lower())
            all_text = re.sub(r'\d+', '', all_text)
            
            words = all_text.split()
            
            filtered_words = [
                word for word in words 
                if word not in stop_words 
                and len(word) > 2
                and word.isalpha()
            ]
            
            word_freq = Counter(filtered_words)
            most_common_words = word_freq.most_common(100)
            filtered_text = ' '.join([word for word, freq in most_common_words])
            
            return filtered_text
    
    return None

def get_sources_stats(company_id=None):
    """Получение статистики по источникам"""
    with engine.connect() as conn:
        if company_id:
            query = text("""
                SELECT 
                    src.name as source,
                    src.type as source_type,
                    COUNT(DISTINCT m.text || '||' || m.author) as review_count
                FROM mentions m
                JOIN sources src ON m.source_id = src.source_id
                JOIN sentiments s ON m.mention_id = s.mention_id
                WHERE s.sentiment IS NOT NULL
                    AND m.company_id = :company_id
                GROUP BY src.name, src.type 
                ORDER BY review_count DESC
            """)
            result = conn.execute(query, {"company_id": company_id})
        else:
            query = text("""
                SELECT 
                    src.name as source,
                    src.type as source_type,
                    COUNT(DISTINCT m.text || '||' || m.author) as review_count
                FROM mentions m
                JOIN sources src ON m.source_id = src.source_id
                JOIN sentiments s ON m.mention_id = s.mention_id
                WHERE s.sentiment IS NOT NULL
                GROUP BY src.name, src.type 
                ORDER BY review_count DESC
            """)
            result = conn.execute(query)
        
        rows = result.fetchall()
        if rows:
            total = sum(row[2] for row in rows)
            df = pd.DataFrame(rows, columns=['source', 'source_type', 'review_count'])
            df['percentage'] = (df['review_count'] / total * 100).round(1)
            return df
    
    return pd.DataFrame()

st.set_page_config(
    page_title="Reputation monitor",
    page_icon="📊",
    layout="wide"
)

st.title("Reputation monitor")
st.markdown("---")

# Инициализация session_state
if 'reviews_loaded' not in st.session_state:
    st.session_state.reviews_loaded = False
if 'company_id_loaded' not in st.session_state:
    st.session_state.company_id_loaded = None

with st.sidebar:
    st.header("⚙️ Параметры мониторинга")
    
    company_name = st.text_input(
        "Название компании или приложения",
        placeholder="Например: VK, Яндекс, Telegram"
    )

    if st.button("🔍 Найти ID приложений автоматически"):
        if company_name:
            with st.spinner("Поиск ID приложений..."):
                gp_id, as_id = get_ids(company_name)
                
                if gp_id:
                    st.success(f"Найден Google Play ID: {gp_id}")
                    st.session_state.gp_id = gp_id
                else:
                    st.warning("Google Play ID не найден")
                
                if as_id:
                    st.success(f"Найден App Store ID: {as_id}")
                    st.session_state.as_id = as_id
                else:
                    st.warning("App Store ID не найден")
        else:
            st.error("Введите название компании")

    col1, col2 = st.columns(2)
    with col1:
        gp_id_final = st.text_input(
            "Google Play ID",
            value=st.session_state.get('gp_id', '')
        )
    with col2:
        as_id_final = st.text_input(
            "App Store ID", 
            value=st.session_state.get('as_id', '')
        )
        
    # Количество отзывов
    count1 = st.slider(
        "Количество отзывов из App Store",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Сколько последних отзывов собрать из App Store (максимум)"
    )

    count2 = st.slider(
        "Количество отзывов из Google Play",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Сколько последних отзывов собрать из Google Play(максимум)"
    )
    
    st.markdown("---")
    
    run_button = st.button(
        "🚀 Запустить мониторинг",
        type="primary",
        use_container_width=True
    )
    
    if st.button("🗑️ Очистить кэш", use_container_width=True):
        st.cache_data.clear()
        st.session_state.reviews_loaded = False
        st.success("Кэш очищен!")
    
    st.markdown("---")
    st.markdown("### 📊 О системе")
    st.info(
        "Система анализирует тональность отзывов из Google Play "
        "и App Store, показывая распределение позитивных, нейтральных "
        "и негативных отзывов."
    )

if run_button:
    if not company_name:
        st.error("❌ Пожалуйста, введите название компании")
    elif not gp_id_final and not as_id_final:
        st.error("❌ Укажите хотя бы один ID приложения")
    else:
        with st.spinner("Сбор и анализ отзывов..."):
            company_id = get_or_create_company(company_name)
            
            total_reviews = 0
            progress_bar = st.progress(0)
            
            # Google Play
            if gp_id_final:
                source_id = get_or_create_source("google_play")
                with st.spinner("Загрузка из Google Play..."):
                    # ВАЖНО: загружаем ровно count отзывов, НЕ count*2
                    df_gp = scrape_google_play(gp_id_final, count2)
                    if not df_gp.empty:
                        # Обрезаем ровно до count
                        if len(df_gp) > count2:
                            df_gp = df_gp.head(count2)
                        # Удаляем дубликаты
                        df_gp = df_gp.drop_duplicates(subset=['text', 'author'], keep='first')
                        # Проверяем существующие
                        df_gp = check_existing_reviews(df_gp, company_id, source_id)
                        
                        if not df_gp.empty:
                            save_mentions(df_gp, company_id, source_id)
                            total_reviews += len(df_gp)
                            st.success(f"✅ Загружено {len(df_gp)} новых отзывов из Google Play")
                        else:
                            st.info("ℹ️ Нет новых отзывов из Google Play")
                    else:
                        st.warning("⚠️ Не удалось загрузить отзывы из Google Play")
                progress_bar.progress(0.5)
            
            # App Store
            if as_id_final:
                source_id = get_or_create_source("app_store")
                with st.spinner("Загрузка из App Store..."):
                    # ВАЖНО: загружаем ровно count отзывов
                    df_as = scrape_app_store_rss_bulk(as_id_final, count1)
                    if not df_as.empty:
                        # Обрезаем ровно до count
                        if len(df_as) > count1:
                            df_as = df_as.head(count1)
                        # Удаляем дубликаты
                        df_as = df_as.drop_duplicates(subset=['text', 'author'], keep='first')
                        # Проверяем существующие
                        df_as = check_existing_reviews(df_as, company_id, source_id)
                        
                        if not df_as.empty:
                            save_mentions(df_as, company_id, source_id)
                            total_reviews += len(df_as)
                            st.success(f"✅ Загружено {len(df_as)} новых отзывов из App Store")
                        else:
                            st.info("ℹ️ Нет новых отзывов из App Store")
                    else:
                        st.warning("⚠️ Не удалось загрузить отзывы из App Store")
                progress_bar.progress(1.0)
            
            progress_bar.empty()
            
            if total_reviews > 0:
                st.balloons()
                st.success(f"🎉 Всего собрано {total_reviews} новых уникальных отзывов!")
                
                with st.spinner("Анализ тональности..."):
                    analyze_reviews_incrementally(batch_size=200)
                    st.success("✅ Анализ завершен!")
                
                st.cache_data.clear()
                st.session_state.reviews_loaded = True
                st.session_state.company_id_loaded = company_id
                st.rerun()
                
            else:
                st.warning("⚠️ Нет новых уникальных отзывов для анализа")
                stats = get_sentiment_stats(company_id)
                if stats and stats['total_analyzed'] > 0:
                    st.info(f"📊 В базе уже есть {stats['total_analyzed']} уникальных отзывов")
                    st.session_state.reviews_loaded = True
                    st.session_state.company_id_loaded = company_id
                else:
                    st.error("❌ Не удалось собрать отзывы")

# Отображение данных
if st.session_state.reviews_loaded or (run_button and 'company_id' in locals()):
    company_id_to_show = st.session_state.get('company_id_loaded', company_id if 'company_id' in locals() else None)
    
    if company_id_to_show:
        stats = get_sentiment_stats(company_id_to_show)
        
        if stats and stats['total_analyzed'] > 0:
            # Круговая диаграмма
            labels = ['Положительные', 'Нейтральные', 'Отрицательные']
            sizes = [stats['positive_count'], stats['neutral_count'], stats['negative_count']]
            colors = ['#2ecc71', '#f39c12', '#e74c3c']

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title(f'Всего уникальных отзывов: {stats["total_analyzed"]}')

            col_pie_chart, col_main_trends, col_pros_cons = st.columns(3)

            with col_pie_chart:
                st.markdown("#### Распределение отзывов")
                st.pyplot(fig)

            with col_main_trends:
                st.markdown("#### Основные тренды")
                total_positive = stats['positive_count']
                total_negative = stats['negative_count']
                total_all = stats['total_analyzed']
                sentiment_ratio = (total_positive - total_negative) / total_all * 100 if total_all > 0 else 0
                
                st.markdown(f"- 📈 Общий тональный индекс: {sentiment_ratio:.1f}%")
                st.markdown(f"- 💬 Положительных: {total_positive} ({total_positive/total_all*100:.1f}%)")
                st.markdown(f"- ⚠️ Отрицательных: {total_negative} ({total_negative/total_all*100:.1f}%)")

            with col_pros_cons:
                st.markdown("#### Плюсы и минусы")
                pros_cons_data = {
                    "Плюсы": [f"Положительные: {total_positive} шт. ({total_positive/total_all*100:.1f}%)"],
                    "Минусы": [f"Отрицательные: {total_negative} шт. ({total_negative/total_all*100:.1f}%)"]
                }
                st.dataframe(pros_cons_data)

            # Динамика
            trends_df = get_daily_trends(company_id_to_show, days=30)
            
            if not trends_df.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(trends_df['review_date'], trends_df['positive_pct'], 
                        label='Положительные', color='#2ecc71', marker='o')
                ax2.plot(trends_df['review_date'], trends_df['negative_pct'], 
                        label='Отрицательные', color='#e74c3c', marker='o')
                ax2.set_xlabel('Дата')
                ax2.set_ylabel('Доля, %')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                
                text_data = get_word_frequencies(company_id_to_show)
                if text_data and len(text_data.strip()) > 0:
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100,
                        min_word_length=3,
                        prefer_horizontal=0.7,
                        relative_scaling=0.5,
                        stopwords=None,
                        collocations=False,
                        random_state=42
                    ).generate(text_data)
                    
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    ax3.imshow(wordcloud, interpolation='bilinear')
                    ax3.axis('off')
                    ax3.set_title('Частотные слова', fontsize=12, pad=10)
                else:
                    fig3 = None
                
                col_dynamic, col_wordcloud = st.columns(2)
                with col_dynamic:
                    st.markdown("#### Динамика тональности")
                    st.pyplot(fig2)
                with col_wordcloud:
                    st.markdown("### Облако слов")
                    if fig3:
                        st.pyplot(fig3)

            st.markdown("---")
            st.markdown("### Детальная информация")

            col_examples, col_sources = st.columns([4, 1])

            with col_sources:
                st.markdown("#### Источники данных")
                sources_df = get_sources_stats(company_id_to_show)
                if not sources_df.empty:
                    st.dataframe(sources_df, use_container_width=True, hide_index=True)

            with col_examples:
                st.markdown("#### Примеры отзывов")
                examples_df = get_recent_reviews(company_id_to_show, limit=1000)
                
                if not examples_df.empty:
                    sentiment_emoji = {
                        'POSITIVE': '🟢 Позитивный',
                        'NEUTRAL': '🟡 Нейтральный', 
                        'NEGATIVE': '🔴 Негативный'
                    }
                    
                    def format_rating(rating):
                        if pd.isna(rating) or rating == 0:
                            return "⭐ Нет оценки"
                        rating_int = int(rating) if rating else 0
                        stars = '⭐' * rating_int
                        empty_stars = '☆' * (5 - rating_int)
                        return f"{stars}{empty_stars} ({rating_int}/5)"
                    
                    examples_df['Тональность'] = examples_df['sentiment'].map(sentiment_emoji)
                    examples_df['Дата'] = pd.to_datetime(examples_df['date']).dt.strftime('%Y-%m-%d')
                    examples_df['Рейтинг'] = examples_df['rating'].apply(format_rating)
                    
                    display_df = examples_df[['text', 'Рейтинг', 'Тональность', 'source', 'Дата']].copy()
                    display_df.columns = ['Текст', '⭐ Рейтинг', 'Тональность', 'Источник', 'Дата']
                    
                    display_df.reset_index(drop=True, inplace=True)
                    display_df.index = display_df.index + 1
                    display_df.index.name = '№'
                    
                    search_term = st.text_input("🔍 Поиск по отзывам", key="search_reviews_unique")
                    
                    filtered_df = display_df
                    if search_term:
                        filtered_df = display_df[display_df['Текст'].str.contains(search_term, case=False, na=False)]
                    
                    st.dataframe(filtered_df, use_container_width=True, height=400)
                    
                    csv = filtered_df.reset_index(drop=True).to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Скачать отзывы (CSV)",
                        data=csv,
                        file_name=f"reviews_{company_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_btn_main"
                    )
                else:
                    st.info("Нет отзывов для отображения")
else:
    # Информация при первом запуске
    st.info("👈 Настройте параметры в боковой панели и нажмите 'Запустить мониторинг'")
    
    # Показываем пример интерфейса
    st.markdown("""
    ### 🎯 Что делает система?
    
    1. **Собирает** уникальные отзывы из Google Play и App Store о выбранной компании
    2. **Анализирует** тональность каждого отзыва с помощью AI-модели
    3. **Визуализирует** результаты в виде графиков и диаграмм
    4. **Показывает** детальную таблицу с фильтрацией и поиском
    
    ### 🚀 Как использовать?
    
    - Введите название компании
    - Нажмите "Найти ID приложений автоматически" или введите ID вручную
    - Укажите количество отзывов (10-500)
    - Нажмите "Запустить мониторинг"
    
    ### 📊 Примеры ID приложений:
    
    - **VK Google Play**: `com.vkontakte.android`
    - **Яндекс Google Play**: `ru.yandex.searchplugin`
    - **OZON Google Play**: `ru.ozon.app.android`
    
    ### 🔧 Дополнительные возможности:
    
                
    - **Поиск по отзывам** - найти нужные комментарии
    - **Скачать CSV** - экспорт отзывов в файл
    """)
