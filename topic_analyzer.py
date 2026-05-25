# topic_analyzer.py
# Модуль для анализа тем и формирования облака слов из отзывов

import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

# Инициализация NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

# Опциональные импорты (необязательные для базовой функциональности)
BERTOPIC_AVAILABLE = False
NATASHA_AVAILABLE = False
GIGACHAT_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    pass

try:
    from natasha import MorphVocab, Doc, Segmenter, NewsEmbedding, NewsMorphTagger
    NATASHA_AVAILABLE = True
except ImportError:
    pass

try:
    from gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    pass


def get_russian_stopwords():
    """Возвращает список русских стоп-слов"""
    russian_stopwords = stopwords.words("russian")
    extra_stopwords = [
        'это', 'этот', 'эта', 'эти', 'этого', 'этому', 'этим', 'этом',
        'весь', 'вся', 'все', 'всё', 'всего', 'всем', 'всеми', 'всех',
        'такой', 'такая', 'такое', 'такие', 'такого', 'такой', 'таких',
        'очень', 'также', 'можно', 'нужно', 'будет', 'есть', 'было', 'была',
        'были', 'был', 'без', 'для', 'или', 'и', 'в', 'на', 'с', 'к', 'у', 'о',
        'об', 'от', 'до', 'по', 'за', 'под', 'над', 'перед', 'при', 'через',
        'между', 'сквозь', 'около', 'возле', 'мимо', 'вдоль', 'поперек'
    ]
    return list(set(russian_stopwords + extra_stopwords))


def topic_coherence(words, texts):
    """
    Вычисляет когерентность темы на основе встречаемости слов в документах
    
    Args:
        words: список ключевых слов темы
        texts: список текстов отзывов
    
    Returns:
        float: коэффициент когерентности
    """
    if not words or not texts:
        return 0.0
    
    score = 0
    count = 0
    
    for doc in texts:
        doc = doc.lower()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[i] in doc and words[j] in doc:
                    score += 1
                count += 1
    
    return score / (count + 1e-9)


def lda_topic_model(df, n_topics=3, n_words=5, max_iter=50):
    """
    LDA (Latent Dirichlet Allocation) тематическое моделирование
    
    Args:
        df: DataFrame с колонкой 'text'
        n_topics: количество тем
        n_words: количество ключевых слов на тему
        max_iter: максимальное количество итераций
    
    Returns:
        dict: результаты моделирования
    """
    start_time = time.time()
    
    texts = df["text"].dropna().astype(str).tolist()
    russian_stopwords = get_russian_stopwords()
    
    vectorizer = CountVectorizer(
        max_df=0.9,
        min_df=2,
        stop_words=russian_stopwords
    )
    
    dtm = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=42
    )
    
    lda.fit(dtm)
    
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = feature_names[top_indices].tolist()
        topics.append(top_words)
    
    coherence_scores = [topic_coherence(topic, texts) for topic in topics]
    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    elapsed_time = time.time() - start_time
    
    return {
        "model": "LDA",
        "topics": topics,
        "coherence": avg_coherence,
        "per_topic_coherence": coherence_scores,
        "time_sec": elapsed_time,
        "n_topics": n_topics
    }


def nmf_topic_model(df, n_topics=3, n_words=5, max_iter=50):
    """
    NMF (Non-negative Matrix Factorization) тематическое моделирование
    
    Args:
        df: DataFrame с колонкой 'text'
        n_topics: количество тем
        n_words: количество ключевых слов на тему
        max_iter: максимальное количество итераций
    
    Returns:
        dict: результаты моделирования
    """
    start_time = time.time()
    
    texts = df["text"].dropna().astype(str).tolist()
    russian_stopwords = get_russian_stopwords()
    
    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=2,
        stop_words=russian_stopwords
    )
    
    dtm = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    nmf = NMF(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=42
    )
    
    nmf.fit(dtm)
    
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = feature_names[top_indices].tolist()
        topics.append(top_words)
    
    doc_topic_dist = nmf.transform(dtm)
    doc_topics = np.argmax(doc_topic_dist, axis=1)
    
    clusters = {i: [] for i in range(n_topics)}
    for text, topic_id in zip(texts, doc_topics):
        clusters[topic_id].append(text)
    
    coherence_scores = [topic_coherence(topic, texts) for topic in topics]
    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    elapsed_time = time.time() - start_time
    
    return {
        "model": "NMF",
        "topics": topics,
        "clusters": clusters,
        "coherence": avg_coherence,
        "per_topic_coherence": coherence_scores,
        "time_sec": elapsed_time,
        "n_topics": n_topics
    }


def bertopic_model(df, n_words=5):
    """
    BERTopic тематическое моделирование (если доступно)
    
    Args:
        df: DataFrame с колонкой 'text'
        n_words: количество ключевых слов на тему
    
    Returns:
        dict: результаты моделирования
    """
    if not BERTOPIC_AVAILABLE:
        return {
            "model": "BERTopic",
            "topics": [],
            "doc_topic_assignment": [],
            "coherence": 0.0,
            "per_topic_coherence": [],
            "time_sec": 0,
            "n_topics": 0,
            "error": "BERTopic not installed"
        }
    
    start_time = time.time()
    texts = df["text"].dropna().astype(str).tolist()
    
    topic_model = BERTopic(
        language="multilingual",
        calculate_probabilities=True,
        verbose=False
    )
    
    topics, probs = topic_model.fit_transform(texts)
    
    unique_topics = set(topics)
    unique_topics.discard(-1)
    
    topic_words = []
    for topic_id in sorted(unique_topics):
        words = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in words[:n_words]] if words else []
        topic_words.append(top_words)
    
    coherence_scores = [topic_coherence(topic, texts) for topic in topic_words]
    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    elapsed_time = time.time() - start_time
    
    return {
        "model": "BERTopic",
        "topics": topic_words,
        "doc_topic_assignment": topics,
        "coherence": avg_coherence,
        "per_topic_coherence": coherence_scores,
        "time_sec": elapsed_time,
        "n_topics": len(topic_words)
    }


def gigachat_summarize_topic(words, credentials=None):
    """
    Суммаризация темы с помощью GigaChat (если доступно)
    
    Args:
        words: список ключевых слов темы
        credentials: учетные данные GigaChat
    
    Returns:
        str: описание темы
    """
    if not GIGACHAT_AVAILABLE:
        return ", ".join(words[:5]) if words else "Тема не распознана"
    
    if not credentials or not words:
        return ", ".join(words[:5]) if words else "Тема не распознана"
    
    try:
        giga = GigaChat(credentials=credentials, verify_ssl_certs=False)
        
        prompt = (
            "Сформулируй короткое осмысленное предложение на русском языке.\n"
            "Это название темы пользовательских отзывов.\n"
            "Максимум 12-15 слов.\n"
            "Сформулируй исключительно предложение без добавления в начале 'отзывы'.\n\n"
            f"Ключевые слова: {', '.join(words)}"
        )
        
        response = giga.chat(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Ты превращаешь ключевые слова в понятные темы."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 60
            }
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"Ошибка GigaChat: {e}")
        return ", ".join(words[:5])


def build_topics_dataframe(model_result, method="NMF + GigaChat", gigachat_creds=None):
    """
    Создает DataFrame с результатами тематического моделирования
    
    Args:
        model_result: результат работы одной из моделей
        method: название метода для поля source
        gigachat_creds: учетные данные GigaChat для суммаризации
    
    Returns:
        DataFrame с колонками: sentence, method, relevance
    """
    rows = []
    
    for i, topic_words in enumerate(model_result["topics"]):
        if topic_words:
            sentence = gigachat_summarize_topic(topic_words, gigachat_creds)
        else:
            sentence = "Пустая тема"
        
        relevance = model_result.get(
            "per_topic_coherence",
            [0.8] * len(model_result["topics"])
        )[i] if model_result["topics"] else 0.0
        
        rows.append({
            "sentence": sentence,
            "method": method,
            "relevance": float(relevance)
        })
    
    return pd.DataFrame(rows)


def get_word_frequencies_simple(df, top_n=50):
    """
    Упрощенное получение частотности слов (без лемматизации, но быстрее)
    
    Args:
        df: DataFrame с колонкой 'text'
        top_n: количество самых частотных словаря
    
    Returns:
        dict: слово -> частота
    """
    try:
        russian_stopwords = set(stopwords.words('russian'))
    except:
        nltk.download('stopwords')
        russian_stopwords = set(stopwords.words('russian'))
    
    extra_stop_words = {
        'это', 'этот', 'эта', 'эти', 'этого', 'этому', 'этим', 'этом',
        'весь', 'вся', 'все', 'всё', 'всего', 'всем', 'всеми', 'всех',
        'такой', 'такая', 'такое', 'такие', 'такого', 'такой', 'таких',
        'очень', 'также', 'можно', 'нужно', 'будет', 'есть', 'было'
    }
    
    stop_words = russian_stopwords.union(extra_stop_words)
    
    all_text = ' '.join(df['text'].dropna().astype(str).tolist())
    all_text = re.sub(r'[^\w\sа-яА-Я]', '', all_text.lower())
    all_text = re.sub(r'\d+', '', all_text)
    
    words = all_text.split()
    
    filtered_words = [
        word for word in words 
        if word not in stop_words 
        and len(word) > 2
        and word.isalpha()
    ]
    
    word_counts = Counter(filtered_words)
    most_common = dict(word_counts.most_common(top_n))
    
    return most_common


def get_word_frequencies_lemmatized(df, top_n=50):
    """
    Получение частотности слов с лемматизацией (более точная версия)
    
    Args:
        df: DataFrame с колонкой 'text'
        top_n: количество самых частотных слов
    
    Returns:
        dict: слово -> частота
    """
    if not NATASHA_AVAILABLE:
        print("Natasha не установлена, используем упрощенную версию")
        return get_word_frequencies_simple(df, top_n)
    
    try:
        segmenter = Segmenter()
        morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
        
        allowed_pos = {'NOUN', 'ADJ', 'VERB'}
        
        text = " ".join(df["text"].dropna().astype(str).tolist())
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        
        important_words = []
        
        for token in doc.tokens:
            if token.pos in allowed_pos:
                token.lemmatize(morph_vocab)
                lemma = token.lemma.lower()
                
                if lemma.isalpha() and len(lemma) > 2:
                    important_words.append(lemma)
        
        word_counts = Counter(important_words)
        most_common = dict(word_counts.most_common(top_n))
        
        return most_common
    except Exception as e:
        print(f"Ошибка при лемматизации: {e}")
        return get_word_frequencies_simple(df, top_n)


def get_word_frequencies_text(text_data, top_n=50):
    """
    Получение частотности слов из текста
    
    Args:
        text_data: строка с текстом
        top_n: количество самых частотных слов
    
    Returns:
        dict: слово -> частота
    """
    try:
        russian_stopwords = set(stopwords.words('russian'))
    except:
        nltk.download('stopwords')
        russian_stopwords = set(stopwords.words('russian'))
    
    extra_stop_words = {
        'это', 'этот', 'эта', 'эти', 'этого', 'этому', 'этим', 'этом',
        'весь', 'вся', 'все', 'всё', 'всего', 'всем', 'всеми', 'всех',
        'такой', 'такая', 'такое', 'такие'
    }
    
    stop_words = russian_stopwords.union(extra_stop_words)
    
    text_data = re.sub(r'[^\w\sа-яА-Я]', '', text_data.lower())
    text_data = re.sub(r'\d+', '', text_data)
    
    words = text_data.split()
    
    filtered_words = [
        word for word in words 
        if word not in stop_words 
        and len(word) > 2
        and word.isalpha()
    ]
    
    word_counts = Counter(filtered_words)
    most_common = dict(word_counts.most_common(top_n))
    
    return most_common


def run_all_models_comparison(df, topic_range=[2, 3, 5, 7, 10]):
    """
    Запускает сравнение всех трех моделей для разных количеств тем
    
    Args:
        df: DataFrame с колонкой 'text'
        topic_range: список количеств тем для тестирования
    
    Returns:
        DataFrame с результатами сравнения
    """
    results = []
    
    for n_topics in topic_range:
        print(f"Running for n_topics = {n_topics}")
        
        lda_res = lda_topic_model(df, n_topics=n_topics, n_words=5)
        results.append({
            "model": "LDA",
            "n_topics": n_topics,
            "coherence": lda_res["coherence"],
            "time": lda_res["time_sec"]
        })
        
        nmf_res = nmf_topic_model(df, n_topics=n_topics, n_words=5)
        results.append({
            "model": "NMF",
            "n_topics": n_topics,
            "coherence": nmf_res["coherence"],
            "time": nmf_res["time_sec"]
        })
        
        if BERTOPIC_AVAILABLE:
            bert_res = bertopic_model(df)
            results.append({
                "model": "BERTopic",
                "n_topics": bert_res["n_topics"],
                "coherence": bert_res["coherence"],
                "time": bert_res["time_sec"]
            })
    
    return pd.DataFrame(results)


def get_best_topics(df, method='nmf', n_topics=5, n_words=6, use_gigachat=False, gigachat_creds=None):
    """
    Удобная функция для получения лучших тем
    
    Args:
        df: DataFrame с колонкой 'text'
        method: метод моделирования ('lda', 'nmf', 'bertopic')
        n_topics: количество тем
        n_words: количество слов на тему
        use_gigachat: использовать ли GigaChat для суммаризации
        gigachat_creds: учетные данные GigaChat
    
    Returns:
        tuple: (DataFrame с темами, результат модели)
    """
    if method == 'lda':
        model_result = lda_topic_model(df, n_topics=n_topics, n_words=n_words)
        method_name = 'LDA'
    elif method == 'nmf':
        model_result = nmf_topic_model(df, n_topics=n_topics, n_words=n_words)
        method_name = 'NMF'
    elif method == 'bertopic':
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic not installed. Install with: pip install bertopic")
        model_result = bertopic_model(df, n_words=n_words)
        method_name = 'BERTopic'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lda', 'nmf', or 'bertopic'")
    
    if use_gigachat and GIGACHAT_AVAILABLE:
        topics_df = build_topics_dataframe(model_result, f"{method_name} + GigaChat", gigachat_creds)
    else:
        rows = []
        for i, topic_words in enumerate(model_result["topics"]):
            sentence = ", ".join(topic_words) if topic_words else "Пустая тема"
            relevance = model_result.get("per_topic_coherence", [0.8] * len(model_result["topics"]))[i] if model_result["topics"] else 0.0
            rows.append({
                "sentence": sentence,
                "method": method_name,
                "relevance": float(relevance)
            })
        topics_df = pd.DataFrame(rows)
    
    return topics_df, model_result


# Для обратной совместимости
get_word_frequencies = get_word_frequencies_simple