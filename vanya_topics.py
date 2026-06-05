import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
import nltk
from bertopic import BERTopic
from natasha import MorphVocab, Doc, Segmenter, NewsEmbedding, NewsMorphTagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import time
from transformers import pipeline
from gigachat import GigaChat

# Loading data

def get_config():
    config_file = "config.txt"
    with open (config_file, 'r') as f:
        SUPABASE_URL = f.readline().strip()
        SUPABASE_KEY = f.readline().strip()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

def import_data(table_name, company_id, output_file="output.csv", size = 3000, batch_size=1000):
    supabase = get_config()
    all_data = []
    start = 0

    while len(all_data) < size:
        try:
            response = supabase.table(table_name).select("mention_id, text").eq("company_id", company_id).range(start, start + batch_size - 1).execute()

            data = response.data

            if not data:
                print(f"Нет данных для company_id={company_id}.")
                break
            
            all_data.extend(data)

            print(f"Сохранено {len(data)} строк (всего: {len(all_data)})")

            if len(data) < batch_size:
                break

            start += batch_size

        except Exception as e:
            print("Ошибка при получении данных:", e)
            print(e)

    if not all_data:
        print("Нет данных")
        return None

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(df)} строк в {output_file}")

    return df
        
# Coherence metric

def topic_coherence(words, texts):

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

# Latent Dirichlet Allocation

def lda_topic_model(df, n_topics=3, n_words=5, max_iter=50):

    start_time = time.time()

    nltk.download("stopwords")
    russian_stopwords = stopwords.words("russian")
    texts = df["text"].dropna().astype(str).tolist()

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

    doc_topic_dist = lda.transform(dtm)
    doc_topics = np.argmax(doc_topic_dist, axis=1)

    coherence_scores = [
        topic_coherence(topic, texts) for topic in topics
    ]

    avg_coherence = float(np.mean(coherence_scores))

    elapsed_time = time.time() - start_time

    result = {
        "model": "LDA",
        "topics": topics,
        "coherence": avg_coherence,
        "per_topic_coherence": coherence_scores,
        "time_sec": elapsed_time,
        "n_topics": n_topics
    }

    return result

# Non-Negative Matrix Factorization

def nmf_topic_model(df, n_topics=3, n_words=5, max_iter=50):

    start_time = time.time()

    nltk.download("stopwords")
    russian_stopwords = stopwords.words("russian")
    texts = df["text"].dropna().astype(str).tolist()

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

    coherence_scores = [
        topic_coherence(topic, texts) for topic in topics
    ]

    avg_coherence = float(np.mean(coherence_scores))

    elapsed_time = time.time() - start_time

    result = {
        "model": "NMF",
        "topics": topics,
        "clusters": clusters,
        "coherence": avg_coherence,
        "per_topic_coherence": coherence_scores,
        "time_sec": elapsed_time,
        "n_topics": n_topics
    }

    return result

# BERTopic model

def bertopic_model(df, n_words=5):

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

        top_words = [w for w, _ in words[:n_words]]
        topic_words.append(top_words)

    doc_topics = topics

    coherence_scores = [
        topic_coherence(topic, texts) for topic in topic_words
    ]

    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0

    elapsed_time = time.time() - start_time

    result = {
        "model": "BERTopic",
        "topics": topic_words,
        "doc_topic_assignment": doc_topics,
        "coherence": avg_coherence,
        "per_topic_coherence": coherence_scores,
        "time_sec": elapsed_time,
        "n_topics": len(topic_words)
    }

    return result

# Summarization

def gigachat_summarize_topic(words):

    if not words:
        return "Неопределённая тема"
    
    giga = GigaChat(
    credentials="MDE5ZTQ2MDAtMzk1MC03MTAwLWJlOGQtZTgyYmNmOTk5MGYwOjMwZDlhY2MwLTllNGItNDZmNy1hY2QwLTE2MjYxYzVlODY5Yw==",
    verify_ssl_certs=False)
    
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

    return response.choices[0].message.content.strip()

def build_topics_dataframe(nmf_result, method="NMF + GigaChat"):
    rows = []

    for i, topic_words in enumerate(nmf_result["topics"]):

        sentence = gigachat_summarize_topic(topic_words)

        relevance = nmf_result.get(
            "per_topic_coherence",
            [0.8] * len(nmf_result["topics"])
        )[i]

        rows.append({
            "sentence": sentence.strip('"'),
            "method": method,
            "relevance": float(relevance)
        })

    return pd.DataFrame(rows)

# Saving results to DB

def upload_to_db(final_topics, company_id):

    df = final_topics.copy()

    df = df.rename(columns={
        "sentence": "trend_text",
        "method": "source"
    })

    df["company_id"] = company_id
    df["trend_type"] = "general"

    if "relevance" not in df.columns:
        df["relevance"] = 0.8

    supabase = get_config()
    supabase.table("topic_summary").delete().eq("company_id", company_id).execute()
    records = df.to_dict(orient="records")

    for r in records:
        data = {
            "company_id": int(r["company_id"]),
            "trend_text": r.get("trend_text", ""),
            "trend_type": r.get("trend_type", "general"),
            "source": r.get("source", "unknown"),
            "relevance": float(r.get("relevance", 0.8))
        }

        try:
            supabase.table("topic_summary").insert(data).execute()

            print("Inserted:", data["trend_text"][:60])

        except Exception as e:
            print("Error:", e, data)

# Wordcloud 

def get_word_frequencies(df, top_n=50):
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    text = " ".join(df["text"].dropna().astype(str).tolist())
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    allowed_pos = {'NOUN', 'ADJ', 'VERB'}
    
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

def upload_wordcloud_to_db(word_freq_dict, company_id):

    supabase = get_config()
    records =[
        {
            "company_id": int(company_id),
            "word": str(word),
            "frequency": int(freq)
        }
        for word, freq in word_freq_dict.items()
    ]
    
    try:
        supabase.table("wordcloud_cache").delete().eq("company_id", company_id).execute()
        response = supabase.table("wordcloud_cache").insert(records).execute()
        print(f"Успешно загружено {len(records)} слов для компании {company_id}")
        
    except Exception as e:
        print(f"Ошибка при загрузке облака слов в БД: {e}")
        return None

# Comparing 3 algorithms

def run_experiment(df, topic_range=[2, 3, 5, 7, 10]):
    
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

        bert_res = bertopic_model(df)

        results.append({
            "model": "BERTopic",
            "n_topics": bert_res["n_topics"],
            "coherence": bert_res["coherence"],
            "time": bert_res["time_sec"]
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    company_id = 8

    giga = GigaChat(
    credentials="MDE5ZTQ2MDAtMzk1MC03MTAwLWJlOGQtZTgyYmNmOTk5MGYwOjMwZDlhY2MwLTllNGItNDZmNy1hY2QwLTE2MjYxYzVlODY5Yw==",
    verify_ssl_certs=False)
    
    df = import_data("mentions", company_id, size = 4000)
    nmf_result = nmf_topic_model(df, n_topics = 6, n_words = 5)
    df_nmf_summ = build_topics_dataframe(nmf_result, method="NMF + GigaChat")
    print(df_nmf_summ)

    upload_to_db(df_nmf_summ, company_id)
    cloud_words_dict = get_word_frequencies(df, 30)
    upload_wordcloud_to_db(cloud_words_dict, company_id)



