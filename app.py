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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from parser import (
    get_ids, get_or_create_company, get_or_create_source, 
    scrape_google_play, scrape_app_store_rss_bulk, save_mentions,
    scrape_google_maps, scrape_vk_newsfeed, scrape_habr_rss, scrape_google_news
)
from reputation_monitor import analyze_reviews_incrementally, engine
from vanya_topics import lda_topic_model, nmf_topic_model, bertopic_model, run_experiment, get_word_frequencies, gigachat_summarize_topic
# Загрузка NLTK данных
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def check_existing_reviews(df, company_id, source_id):
    """Проверяет, какие отзывы уже есть в БД"""
    if df.empty:
        return df
    
    with engine.connect() as conn:
        query = text("""
            SELECT DISTINCT text, author, date 
            FROM mentions 
            WHERE company_id = :company_id AND source_id = :source_id
        """)
        existing = conn.execute(query, {"company_id": company_id, "source_id": source_id}).fetchall()
        
        existing_set = set()
        for row in existing:
            text_normalized = ' '.join(row[0].strip().split()) if row[0] else ''
            author_normalized = row[1].strip() if row[1] else ''
            date_normalized = row[2] if row[2] else None
            existing_set.add((text_normalized, author_normalized, date_normalized))
    
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

# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С БД ====================

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
                JOIN mentions m ON s.mention_id = m.mention_id
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

# ==================== ФУНКЦИИ ДЛЯ ТЕМАТИЧЕСКОГО АНАЛИЗА ====================
# ==================== ФУНКЦИИ ДЛЯ ТЕМАТИЧЕСКОГО АНАЛИЗА ====================
def display_topic_analysis(df, company_name):
    """Отображение тематического анализа отзывов с интеграцией GigaChat"""
    
    if df.empty:
        st.info("Нет данных для тематического анализа")
        return
    
    st.markdown("### 🎯 Ключевые темы и тренды")
    st.markdown("Искусственный интеллект сгруппировал отзывы по смыслу и обобщил.")
    
    # Создаем вкладки для разных методов анализа
    tab_lda, tab_nmf, tab_bertopic, tab_comparison = st.tabs([
        "📚 LDA (Классический разбор)", 
        "⚡ NMF (Экспресс-анализ)", 
        "🧠 BERTopic (Продвинутая нейросеть)", 
        "📊 Сравнение алгоритмов"
    ])
    
    with tab_lda:
        st.markdown("#### LDA (Классический тематический разбор)")
        st.caption("Проверенный временем метод. Отлично раскладывает большие объемы отзывов на понятные составляющие.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            n_topics_lda = st.slider(
                "Сколько тем выделить?",
                min_value=2, max_value=8, value=3, step=1, key="lda_topics"
            )
        
        with col1:
            if st.button("🔍 Запустить классический анализ", key="run_lda"):
                with st.spinner("Алгоритм группирует отзывы..."):
                    try:
                        lda_result = lda_topic_model(df, n_topics=n_topics_lda, n_words=6)
                        
                        if lda_result and lda_result['topics']:
                            st.success(f"✅ Успешно выделено {len(lda_result['topics'])} ключевых направлений!")
                            
                            # Понятные метрики вместо сухих терминов
                            col_coh, col_time = st.columns(2)
                            with col_coh:
                                st.metric("🎯 Четкость разделения тем", f"{lda_result['coherence']:.2f}")
                                st.caption("👉 Чем выше этот показатель, тем более обособленными и логичными получились группы отзывов.")
                            with col_time:
                                st.metric("⏱️ Время работы алгоритма", f"{lda_result['time_sec']:.2f} сек")
                            
                            st.markdown("#### 📌 Что обсуждают пользователи (Расшифровка ИИ):")
                            
                            for idx, (topic_words, coherence) in enumerate(zip(lda_result['topics'], lda_result['per_topic_coherence'])):
                                # Запускаем GigaChat для формулирования темы
                                with st.spinner(f"Генерация понятного названия для Темы №{idx + 1}..."):
                                    try:
                                        human_title = gigachat_summarize_topic(topic_words)
                                    except Exception:
                                        human_title = f"Тема №{idx + 1}: Набор ключевых слов"
                                
                                with st.expander(f"📋 {human_title} (Четкость темы: {coherence:.2f})"):
                                    st.markdown("**Ключевые слова, по которым нейросеть узнала тему:**")
                                    st.caption(", ".join([f"`{w}`" for w in topic_words]))
                                    
                                    # Примеры отзывов
                                    st.markdown("**📝 Примеры отзывов из этой группы:**")
                                    examples_found = 0
                                    for text in df['text'].head(100):
                                        if any(word in str(text).lower() for word in topic_words):
                                            st.text(f"• {str(text)[:150]}...")
                                            examples_found += 1
                                            if examples_found >= 3:
                                                break
                                    if examples_found == 0:
                                        st.caption("Прямых примеров текста не найдено.")
                        else:
                            st.warning("Не удалось выделить темы")
                    except Exception as e:
                        st.error(f"Ошибка при анализе: {str(e)}")
    
    with tab_nmf:
        st.markdown("#### NMF (Экспресс-анализ)")
        st.caption("Математический метод, идеально подходящий для коротких текстов и емких отзывов из магазинов приложений.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            n_topics_nmf = st.slider(
                "Сколько тем выделить?",
                min_value=2, max_value=8, value=3, step=1, key="nmf_topics"
            )
        
        with col1:
            if st.button("⚡ Запустить экспресс-анализ", key="run_nmf"):
                with st.spinner("Быстрый анализ текстов..."):
                    try:
                        nmf_result = nmf_topic_model(df, n_topics=n_topics_nmf, n_words=6)
                        
                        if nmf_result and nmf_result['topics']:
                            st.success(f"✅ Выделено {len(nmf_result['topics'])} тем")
                            
                            col_coh, col_time = st.columns(2)
                            with col_coh:
                                st.metric("🎯 Четкость разделения тем", f"{nmf_result['coherence']:.2f}")
                            with col_time:
                                st.metric("⏱️ Время работы алгоритма", f"{nmf_result['time_sec']:.2f} сек")
                            
                            st.markdown("#### 📌 Результаты анализа от GigaChat:")
                            
                            for idx, (topic_words, coherence) in enumerate(zip(nmf_result['topics'], nmf_result['per_topic_coherence'])):
                                with st.spinner(f"Формулирование сути Темы №{idx + 1}..."):
                                    try:
                                        human_title = gigachat_summarize_topic(topic_words)
                                    except Exception:
                                        human_title = f"Тема №{idx + 1}"
                                        
                                with st.expander(f"📋 {human_title} (Четкость: {coherence:.2f})"):
                                    st.markdown("**Связанные маркеры:** " + " ".join([f"`{word}`" for word in topic_words]))
                                    
                                    if 'clusters' in nmf_result and idx in nmf_result['clusters']:
                                        examples = nmf_result['clusters'][idx][:3]
                                        if examples:
                                            st.markdown("**📝 Что пишут люди:**")
                                            for example in examples:
                                                st.caption(f"• {example[:200]}...")
                                    else:
                                        st.markdown("**📝 Примеры отзывов:**")
                                        examples_found = 0
                                        for text in df['text']:
                                            if any(word in str(text).lower() for word in topic_words):
                                                st.text(f"• {str(text)[:150]}...")
                                                examples_found += 1
                                                if examples_found >= 3:
                                                    break
                        else:
                            st.warning("Не удалось выделить темы")
                    except Exception as e:
                        st.error(f"Ошибка при NMF анализе: {str(e)}")
    
    with tab_bertopic:
        st.markdown("#### BERTopic (Продвинутый нейросетевой анализ)")
        st.caption("Самый умный подход. Использует языковые модели-трансформеры для улавливания тонкого контекста и сарказма.")
        
        bertopic_available = False
        try:
            from bertopic import BERTopic
            bertopic_available = True
        except ImportError:
            st.warning("⚠️ Компонент BERTopic не установлен на сервере.")
        
        if bertopic_available:
            col1, col2 = st.columns([2, 1])
            with col2:
                n_words_bert = st.slider("Глубина анализа темы (слов)", min_value=3, max_value=10, value=5, key="bert_words")
            
            with col1:
                if st.button("🧠 Включить нейросетевой поиск", key="run_bertopic"):
                    with st.spinner("Нейросеть глубоко изучает тексты (это может занять до пары минут)..."):
                        try:
                            bert_result = bertopic_model(df, n_words=n_words_bert)
                            
                            if bert_result and bert_result['topics'] and bert_result['n_topics'] > 0:
                                st.success(f"✅ Нейросеть выявила {bert_result['n_topics']} устойчивых тем")
                                
                                col_coh, col_time = st.columns(2)
                                with col_coh:
                                    st.metric("🎯 Качество понимания контекста", f"{bert_result['coherence']:.2f}")
                                with col_time:
                                    st.metric("⏱️ Время размышления ИИ", f"{bert_result['time_sec']:.2f} сек")
                                
                                st.markdown("#### 📌 Темы, найденные трансформером:")
                                for idx, topic_words in enumerate(bert_result['topics']):
                                    coherence = bert_result['per_topic_coherence'][idx] if idx < len(bert_result['per_topic_coherence']) else 0.5
                                    
                                    with st.spinner(f"GigaChat переводит Тему №{idx + 1}..."):
                                        try:
                                            human_title = gigachat_summarize_topic(topic_words)
                                        except Exception:
                                            human_title = f"Группа смыслов №{idx + 1}"
                                            
                                    with st.expander(f"🧠 {human_title} (Качество смысловой группы: {coherence:.2f})"):
                                        st.markdown("**Обнаруженные ИИ взаимосвязи:** " + ", ".join([f"`{w}`" for w in topic_words]))
                                        
                                        st.markdown("**📝 Конкретные отзывы из этой категории:**")
                                        examples_found = 0
                                        for i, text in enumerate(df['text'].head(200)):
                                            if 'doc_topic_assignment' in bert_result and i < len(bert_result['doc_topic_assignment']) and bert_result['doc_topic_assignment'][i] == idx:
                                                st.text(f"• {str(text)[:150]}...")
                                                examples_found += 1
                                                if examples_found >= 3:
                                                    break
                                            elif any(word.lower() in str(text).lower() for word in topic_words):
                                                st.text(f"• {str(text)[:150]}...")
                                                examples_found += 1
                                                if examples_found >= 3:
                                                    break
                            else:
                                st.warning("Нейросеть не смогла сгруппировать текущий объем данных. Попробуйте загрузить больше отзывов.")
                        except Exception as e:
                            st.error(f"Ошибка ИИ-анализа: {str(e)}")
        
    
    with tab_comparison:
        st.markdown("#### Сравнение математической точности алгоритмов")
        st.caption("Вкладка для проверки того, какой математический метод сработает чище всего на ваших данных.")
        
        if st.button("📊 Запустить соревнование моделей", key="run_comparison"):
            with st.spinner("Тестирование моделей..."):
                try:
                    topic_range = [2, 3, 5, 7]
                    comparison_df = run_experiment(df, topic_range)
                    
                    if not comparison_df.empty:
                        st.success("✅ Соревнование завершено!")
                        
                        fig_comp = px.line(
                            comparison_df, x='n_topics', y='coherence', color='model',
                            title='Какая модель построила темы наиболее четко?',
                            labels={'n_topics': 'Количество тем в тесте', 'coherence': 'Понятность и четкость группы'},
                            markers=True
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        st.dataframe(
                            comparison_df.round(4), use_container_width=True, hide_index=True,
                            column_config={
                                'model': 'Алгоритм / Модель',
                                'n_topics': 'Количество тем',
                                'coherence': 'Четкость (выше — лучше)',
                                'time': 'Скорость работы (сек)'
                            }
                        )
                        
                        best_model = comparison_df.loc[comparison_df['coherence'].idxmax()]
                        st.markdown(f"🏆 **Победитель тестирования:** Модель **{best_model['model']}** при разбиении на **{int(best_model['n_topics'])}** тем(ы).")
                    else:
                        st.warning("Не удалось собрать данные сравнения.")
                except Exception as e:
                    st.error(f"Ошибка расчета: {str(e)}")


def display_key_topics_summary(df):
    """Отображение краткой сводки по ключевым темам на главной странице с развернутыми предложениями от GigaChat"""
    
    if df.empty:
        return
    
    st.markdown("### 🔥 Главные темы в отзывах по версии GigaChat")
    
    with st.spinner("GigaChat анализирует группы слов и составляет описание тем..."):
        try:
            # Используем алгоритм NMF для выделения 3 ключевых тем
            nmf_result = nmf_topic_model(df, n_topics=3, n_words=5)
            
            if nmf_result and nmf_result['topics']:
                cols = st.columns(min(len(nmf_result['topics']), 3))
                
                for idx, (col, topic_words) in enumerate(zip(cols, nmf_result['topics'])):
                    with col:
                        coherence = nmf_result['per_topic_coherence'][idx]
                        color = "🟢" if coherence > 0.6 else "🟡" if coherence > 0.3 else "🔴"
                        
                        # Обращаемся к GigaChat, чтобы он сформулировал полноценные предложения
                        try:
                            gigachat_sentences = gigachat_summarize_topic(topic_words)
                        except Exception:
                            gigachat_sentences = "Не удалось построить описание. Проверьте подключение к GigaChat."
                        
                        # Красивая карточка, где есть и "Тема X", и предложения ИИ, и маркеры слов
                        st.markdown(f"""
                        <div style="
                            padding: 20px;
                            border-radius: 12px;
                            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                            color: white;
                            margin: 10px 0;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            min-height: 240px;
                            display: flex;
                            flex-direction: column;
                            justify-content: space-between;
                        ">
                            <div>
                                <h4 style="color: white; margin: 0 0 10px 0; font-size: 18px;">
                                    {color} Тема {idx + 1}
                                </h4>
                                <p style="font-size: 14px; margin: 0 0 15px 0; line-height: 1.4; font-weight: 400;">
                                    {gigachat_sentences}
                                </p>
                            </div>
                            <div>
                                <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;">
                                <p style="font-size: 12px; margin: 5px 0; opacity: 0.9;">
                                    <strong>Набор слов:</strong> {', '.join(topic_words)}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
        except Exception as e:
            st.caption(f"Временные технические ограничения при экспресс-анализе тем: {str(e)}")


# ==================== ФУНКЦИИ ДЛЯ СБОРА ДАННЫХ ====================

def scrape_source_with_progress(source_config, company_id, progress_bar, status_text):
    """
    Универсальная функция для сбора данных из источника с отображением прогресса
    """
    source_name = source_config['name']
    scrape_func = source_config['func']
    count = source_config['count']
    
    status_text.text(f"🔄 Загрузка из {source_name}...")
    
    try:
        # Вызов функции парсинга
        if source_name in ['Google Play', 'App Store']:
            df = scrape_func(source_config['id'], count)
        else:
            df = scrape_func(company_name, count) if 'company_name' in source_config else scrape_func(source_config.get('query', company_name), count)
        
        if df is not None and not df.empty:
            # Обрезаем до нужного количества
            if len(df) > count:
                df = df.head(count)
            
            # Удаляем дубликаты
            df = df.drop_duplicates(subset=['text', 'author'], keep='first')
            
            # Получаем source_id
            source_id = get_or_create_source(source_name.lower().replace(' ', '_'))
            
            # Проверяем существующие отзывы
            df = check_existing_reviews(df, company_id, source_id)
            
            if not df.empty:
                # Сохраняем в БД
                save_mentions(df, company_id, source_id)
                status_text.text(f"✅ {source_name}: загружено {len(df)} новых отзывов")
                return len(df)
            else:
                status_text.text(f"ℹ️ {source_name}: нет новых отзывов")
                return 0
        else:
            status_text.text(f"⚠️ {source_name}: не удалось загрузить отзывы")
            return 0
            
    except Exception as e:
        status_text.text(f"❌ {source_name}: ошибка - {str(e)[:50]}")
        return 0
    finally:
        progress_bar.progress(1.0)

# ==================== НАСТРОЙКА STRREAMLIT ====================

st.set_page_config(
    page_title="Reputation Monitor - Полный мониторинг репутации",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Reputation Monitor")
st.markdown("### Комплексный мониторинг репутации компании из 6 источников")
st.markdown("---")

# Инициализация session_state
if 'reviews_loaded' not in st.session_state:
    st.session_state.reviews_loaded = False
if 'company_id_loaded' not in st.session_state:
    st.session_state.company_id_loaded = None
if 'last_company_name' not in st.session_state:
    st.session_state.last_company_name = ""

# ==================== БОКОВАЯ ПАНЕЛЬ ====================

with st.sidebar:
    st.header("⚙️ Параметры мониторинга")
    
    # Ввод названия компании
    company_name = st.text_input(
        "🏢 Название компании или приложения",
        placeholder="Например: VK, Яндекс, OZON",
        value=st.session_state.last_company_name
    )
    
    # Автоматический поиск ID
    if st.button("🔍 Найти ID приложений автоматически", use_container_width=True):
        if company_name:
            with st.spinner("Поиск ID приложений..."):
                gp_id, as_id = get_ids(company_name)
                
                if gp_id or as_id:
                    st.success(f"✅ Найдены ID!")
                    st.session_state.gp_id = gp_id
                    st.session_state.as_id = as_id
                if not gp_id:
                    st.warning("⚠️ Google Play ID не найден")
                
                else:
                    st.warning("⚠️ App Store ID не найден")
        else:
            st.error("Введите название компании")
    
    st.markdown("---")
    
    # Выбор источников
    st.subheader("📡 Источники данных")
    
    # Определение источников с их настройками
    sources_config = {
        "Google Play": {
            "enabled": st.checkbox("📱 Google Play", value=True),
            "id_key": "gp_id_final",
            "default_count": 100,
            "min_count": 10,
            "max_count": 500,
            "requires_id": True
        },
        "App Store": {
            "enabled": st.checkbox("🍎 App Store", value=True),
            "id_key": "as_id_final",
            "default_count": 100,
            "min_count": 10,
            "max_count": 500,
            "requires_id": True
        },
        "Google Карты": {
            "enabled": st.checkbox("🗺️ Google Карты", value=False),
            "default_count": 20,
            "min_count": 5,
            "max_count": 100,
            "requires_id": False
        },
        "ВКонтакте": {
            "enabled": st.checkbox("📰 ВКонтакте", value=False),
            "default_count": 50,
            "min_count": 10,
            "max_count": 200,
            "requires_id": False
        },
        "Habr": {
            "enabled": st.checkbox("💻 Habr", value=False),
            "default_count": 30,
            "min_count": 10,
            "max_count": 100,
            "requires_id": False
        },
        "Google Новости": {
            "enabled": st.checkbox("📰 Google Новости", value=False),
            "default_count": 30,
            "min_count": 10,
            "max_count": 100,
            "requires_id": False
        }
    }
    
    st.markdown("---")
    
    # Ползунки для количества отзывов по каждому источнику
    st.subheader("🔢 Количество отзывов")
    
    source_counts = {}
    for source_name, config in sources_config.items():
        if config["enabled"]:
            count = st.slider(
                f"{source_name}",
                min_value=config["min_count"],
                max_value=config["max_count"],
                value=config["default_count"],
                step=10,
                key=f"count_{source_name}"
            )
            source_counts[source_name] = count
    
    st.markdown("---")
    
    # Кнопка запуска
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
        "Система анализирует тональность упоминаний из Google Play, "
        "App Store, Google Карт, ВКонтакте, Habr и Google Новостей."
    )

# ==================== ОСНОВНАЯ ЛОГИКА ====================

# Сохраняем название компании в session_state
if company_name:
    st.session_state.last_company_name = company_name

if run_button:
    if not company_name:
        st.error("❌ Пожалуйста, введите название компании")
    else:
        # Проверяем, что выбран хотя бы один источник
        enabled_sources = [name for name, config in sources_config.items() if config["enabled"]]
        
        if not enabled_sources:
            st.error("❌ Выберите хотя бы один источник данных")
            st.stop()
        
        # Создаем или получаем компанию
        with st.spinner("Подготовка к сбору данных..."):
            company_id = get_or_create_company(company_name)
        
        total_reviews = 0
        source_results = {}
        
        # Создаем контейнеры для прогресс-баров
        st.markdown("### 📥 Сбор данных из источников")
        
        # Словарь для хранения статусов
        status_containers = {}
        progress_containers = {}
        
        # Создаем UI для каждого источника
        for source_name in enabled_sources:
            col1, col2 = st.columns([3, 1])
            with col1:
                status_containers[source_name] = st.empty()
            with col2:
                progress_containers[source_name] = st.progress(0)
            st.markdown("---")
        
        # Сбор данных из каждого источника
        for source_name in enabled_sources:
            config = sources_config[source_name]
            count = source_counts[source_name]
            
            status_text = status_containers[source_name]
            progress_bar = progress_containers[source_name]
            
            # Обновляем прогресс
            progress_bar.progress(0.2)
            status_text.text(f"🔄 Инициализация {source_name}...")
            
            # Определяем функцию парсинга
            if source_name == "Google Play":
                scrape_func = scrape_google_play
                # Берем ID из session_state, который был найден автоматически
                source_id_value = st.session_state.get('gp_id', '')
                if not source_id_value:
                    status_text.text(f"⚠️ {source_name}: ID не найден")
                    result_count = 0
                else:
                    result_count = scrape_source_with_progress(
                        {"name": source_name, "func": scrape_func, "count": count, "id": source_id_value},
                        company_id, progress_bar, status_text
                    )
                
            elif source_name == "App Store":
                scrape_func = scrape_app_store_rss_bulk
                source_id_value = st.session_state.get('as_id', '')
                if not source_id_value:
                    status_text.text(f"⚠️ {source_name}: ID не найден")
                    result_count = 0
                else:
                    result_count = scrape_source_with_progress(
                        {"name": source_name, "func": scrape_func, "count": count, "id": source_id_value},
                        company_id, progress_bar, status_text
                    )
                
            elif source_name == "Google Карты":
                scrape_func = scrape_google_maps
                result_count = scrape_source_with_progress(
                    {"name": source_name, "func": scrape_func, "count": count, "company_name": company_name},
                    company_id, progress_bar, status_text
                )
                
            elif source_name == "ВКонтакте":
                scrape_func = scrape_vk_newsfeed
                result_count = scrape_source_with_progress(
                    {"name": source_name, "func": scrape_func, "count": count, "query": company_name},
                    company_id, progress_bar, status_text
                )
                
            elif source_name == "Habr":
                scrape_func = scrape_habr_rss
                result_count = scrape_source_with_progress(
                    {"name": source_name, "func": scrape_func, "count": count, "query": company_name},
                    company_id, progress_bar, status_text
                )
                
            elif source_name == "Google Новости":
                scrape_func = scrape_google_news
                result_count = scrape_source_with_progress(
                    {"name": source_name, "func": scrape_func, "count": count, "query": company_name},
                    company_id, progress_bar, status_text
                )
            
            source_results[source_name] = result_count
            total_reviews += result_count
            progress_bar.progress(1.0)
        
        # Отображение итогов сбора
        st.markdown("### 📊 Итоги сбора данных")
        
        cols = st.columns(min(len(enabled_sources), 4))
        for idx, (source_name, count) in enumerate(source_results.items()):
            with cols[idx % 4]:
                if count > 0:
                    st.metric(source_name, f"+{count}", "новых отзывов")
                else:
                    st.metric(source_name, "0", "нет новых отзывов")
        
        if total_reviews > 0:
            st.balloons()
            st.success(f"🎉 Всего собрано {total_reviews} новых упоминаний!")
            
            # Анализ тональности
            st.markdown("### 🔬 Анализ тональности отзывов")
            analyze_progress = st.progress(0)
            analyze_status = st.empty()
            
            
            # Анализируем инкрементально
            with st.spinner("Анализ тональности..."):
                analyze_reviews_incrementally(batch_size=200)
            
            analyze_progress.progress(1.0)
            analyze_status.text("✅ Анализ тональности завершен!")
            
            # Очищаем кэш и обновляем состояние
            st.cache_data.clear()
            st.session_state.reviews_loaded = True
            st.session_state.company_id_loaded = company_id
            
            # Небольшая задержка перед перезагрузкой
            time.sleep(1)
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

# ==================== ОТОБРАЖЕНИЕ ДАННЫХ ====================

if st.session_state.reviews_loaded or (run_button and 'company_id' in locals()):
    company_id_to_show = st.session_state.get('company_id_loaded', company_id if 'company_id' in locals() else None)
    
    if company_id_to_show:
        stats = get_sentiment_stats(company_id_to_show)
        
        if stats and stats['total_analyzed'] > 0:
            # Верхняя панель с метриками
            st.markdown("### 📈 Общая статистика")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Всего отзывов", stats['total_analyzed'])
            with col2:
                pos_pct = (stats['positive_count'] / stats['total_analyzed'] * 100) if stats['total_analyzed'] > 0 else 0
                st.metric("😊 Положительные", f"{stats['positive_count']} ({pos_pct:.1f}%)", delta="+", delta_color="normal")
            with col3:
                neg_pct = (stats['negative_count'] / stats['total_analyzed'] * 100) if stats['total_analyzed'] > 0 else 0
                st.metric("😞 Отрицательные", f"{stats['negative_count']} ({neg_pct:.1f}%)", delta="-", delta_color="inverse")
            with col4:
                sentiment_ratio = (stats['positive_count'] - stats['negative_count']) / stats['total_analyzed'] * 100 if stats['total_analyzed'] > 0 else 0
                st.metric("📈 Тональный индекс", f"{sentiment_ratio:.1f}%")
            
            st.markdown("---")
            
            # Графики
            col_pie_chart, col_trends = st.columns(2)
            
            with col_pie_chart:
                st.markdown("#### Распределение тональности")
                labels = ['Положительные', 'Нейтральные', 'Отрицательные']
                sizes = [stats['positive_count'], stats['neutral_count'], stats['negative_count']]
                colors = ['#2ecc71', '#f39c12', '#e74c3c']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Всего уникальных отзывов: {stats["total_analyzed"]}')
                st.pyplot(fig)
            
            with col_trends:
                trends_df = get_daily_trends(company_id_to_show, days=30)
                if not trends_df.empty:
                    st.markdown("#### Динамика тональности")
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.plot(trends_df['review_date'], trends_df['positive_pct'], 
                            label='Положительные', color='#2ecc71', marker='o', linewidth=2)
                    ax2.plot(trends_df['review_date'], trends_df['negative_pct'], 
                            label='Отрицательные', color='#e74c3c', marker='o', linewidth=2)
                    ax2.set_xlabel('Дата')
                    ax2.set_ylabel('Доля, %')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)
            
            st.markdown("---")

            # Облако слов и тематический анализ
            st.markdown("### 📊 Визуализация и анализ")

            # Создаем вкладки для визуализации
            tab_cloud, tab_topics = st.tabs(["☁️ Облако слов", "🎯 Ключевые темы"])

            with tab_cloud:
                st.markdown("#### Облако ключевых слов")
                # Получаем отзывы для облака слов
                cloud_df = get_recent_reviews(company_id_to_show, limit=500)
                
                if not cloud_df.empty:
                    with st.spinner("Анализ текста и генерация облака слов (лемматизация)..."):
                        try:
                            # Используем вашу функцию из vanya_topics для лемматизации
                            word_freq = get_word_frequencies(cloud_df, top_n=100)
                            
                            if word_freq:
                                # Дополнительно убираем слова, которые часто встречаются, но не несут пользы
                                custom_stopwords = {'приложение', 'очень', 'весь', 'свой', 'который', 'просто', 'это'}
                                for sw in custom_stopwords:
                                    word_freq.pop(sw, None)
                                
                                # Генерируем облако слов на основе частотного словаря
                                wordcloud = WordCloud(
                                    width=800, 
                                    height=400,
                                    background_color='white',
                                    colormap='viridis',
                                    max_words=100,
                                    contour_width=1,
                                    contour_color='steelblue'
                                ).generate_from_frequencies(word_freq)
                                
                                # Отрисовка графика
                                fig_cloud, ax_cloud = plt.subplots(figsize=(10, 5))
                                ax_cloud.imshow(wordcloud, interpolation='bilinear')
                                ax_cloud.axis('off')
                                
                                st.pyplot(fig_cloud)
                            else:
                                st.warning("Не удалось выделить ключевые слова для облака.")
                                
                        except Exception as e:
                            st.error(f"Ошибка при создании облака слов: {str(e)}")
                            st.info("Убедитесь, что установлена библиотека natasha.")
                else:
                    st.info("Нет данных для создания облака слов")

            with tab_topics:
                # Отображаем ключевые темы
                topics_df = get_recent_reviews(company_id_to_show, limit=500)
                if not topics_df.empty:
                    # Показываем краткую сводку
                    display_key_topics_summary(topics_df)
                    
                    # Полный анализ
                    display_topic_analysis(topics_df, company_name)
                else:
                    st.info("Нет данных для тематического анализа")

            st.markdown("---")
            
            # Детальная информация
            st.markdown("### 📋 Детальная информация")
            
            col_examples, col_sources = st.columns([3, 1])
            
            with col_sources:
                st.markdown("#### 📊 Источники данных")
                sources_df = get_sources_stats(company_id_to_show)
                if not sources_df.empty:
                    st.dataframe(sources_df, use_container_width=True, hide_index=True)
                    
                    # Визуализация источников
                    fig4, ax4 = plt.subplots(figsize=(6, 4))
                    ax4.barh(sources_df['source'], sources_df['review_count'], color='skyblue')
                    ax4.set_xlabel('Количество отзывов')
                    ax4.set_title('Отзывы по источникам')
                    st.pyplot(fig4)
            
            with col_examples:
                st.markdown("#### 📝 Примеры отзывов")
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
                    
                    search_term = st.text_input("🔍 Поиск по отзывам", key="search_reviews_main")
                    
                    filtered_df = display_df
                    if search_term:
                        filtered_df = display_df[display_df['Текст'].str.contains(search_term, case=False, na=False)]
                    
                    st.dataframe(filtered_df, use_container_width=True, height=400)
                    
                    # Кнопка скачивания
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
    
    1. **Собирает** уникальные упоминания из **6 источников**:
       - 📱 Google Play
       - 🍎 App Store
       - 🗺️ Google Карты
       - 📰 ВКонтакте
       - 💻 Habr
       - 📰 Google Новости
    
    2. **Анализирует** тональность каждого упоминания с помощью AI-модели
    
    3. **Визуализирует** результаты в виде графиков, диаграмм и облака слов
    
    4. **Показывает** детальную таблицу с фильтрацией и поиском
    
    ### 🚀 Как использовать?
    
    - Введите название компании
    - Нажмите "Найти ID приложений автоматически"
    - Выберите нужные источники в боковой панели
    - Настройте количество отзывов для каждого источника
    - Нажмите "Запустить мониторинг"
    
    
    ### 🔧 Дополнительные возможности:
    
    - **Поиск по отзывам** - найти нужные комментарии
    - **Скачать CSV** - экспорт отзывов в файл
    - **Выбор источников** - гибкая настройка сбора данных
    """)
