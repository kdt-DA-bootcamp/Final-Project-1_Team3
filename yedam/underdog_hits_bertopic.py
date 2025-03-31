import os
import ast
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

load_dotenv()

# ------------------- DB 불러오기 -------------------
def load_data():
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    database = os.getenv('DB_NAME')
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
    videos_df = pd.read_sql("SELECT * FROM video", engine)
    channels_df = pd.read_sql("SELECT * FROM channel", engine)
    return pd.merge(videos_df, channels_df, on="channelID", how="left")

# ------------------- 모델 학습 -------------------
def train_model(df):
    features = ['subscriberCount', 'videosCount', 'duration', 'likeCount', 'commentCount', 'categoryID']
    df = df[features + ['viewCount']].dropna()
    X = df[features]
    y = np.log1p(df['viewCount'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# ------------------- Gap 예측 -------------------
def predict_gap(model, df):
    df = df.copy()
    df['videosCount'] = df.groupby('channelID')['videoID'].transform('count')
    df['duration'] = 300
    df['categoryID'] = 0
    features = ['subscriberCount', 'videosCount', 'duration', 'likeCount', 'commentCount', 'categoryID']
    X = df[features].fillna(0)
    df['predicted_viewCount'] = np.expm1(model.predict(X))
    df['gap'] = df['viewCount'] - df['predicted_viewCount']
    df['gap_ratio'] = df['viewCount'] / (df['predicted_viewCount'] + 1)
    return df

# ------------------- 텍스트 전처리 -------------------
def clean_text(text):
    text = str(text).replace("#", "").replace(",", "").replace(".", "").lower()
    return text

# ------------------- TF-IDF 기반 키워드 추출 -------------------
def tfidf_keywords(texts):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).flatten()
    keywords = sorted(zip(feature_names, scores), key=lambda x: -x[1])
    return [{'keyword': k, 'score': round(v, 4)} for k, v in keywords[:20]]

# ------------------- BERTopic 기반 키워드 추출 -------------------
def bertopic_keywords(texts):
    model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device='cuda')
    topic_model = BERTopic(embedding_model=model, language="multilingual", verbose=True)
    topics, _ = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    keywords = []
    for topic in topic_info.Topic:
        if topic != -1:
            words = topic_model.get_topic(topic)
            keywords.extend([{'keyword': word, 'score': round(weight * 100, 2)} for word, weight in words])
    return keywords

# ------------------- 교차 키워드 계산 -------------------
def calculate_intersection(tfidf, bert):
    tfidf_set = set([k['keyword'] for k in tfidf])
    bert_set = set([k['keyword'] for k in bert])
    intersection = tfidf_set.intersection(bert_set)
    
    # 가중치 추가
    intersection_keywords = []
    for keyword in intersection:
        tfidf_score = next((item['score'] for item in tfidf if item['keyword'] == keyword), 0)
        bert_score = next((item['score'] for item in bert if item['keyword'] == keyword), 0)
        combined_score = tfidf_score * 0.5 + bert_score * 0.5  # 가중 평균
        intersection_keywords.append({'keyword': keyword, 'score': round(combined_score, 4)})
    return intersection_keywords

# ------------------- 전체 분석 -------------------
def analyze():
    df = load_data()
    model = train_model(df)

    # 중형 채널 필터링
    low = df['subscriberCount'].quantile(0.3)
    high = df['subscriberCount'].quantile(0.7)
    df = df[(df['subscriberCount'] > low) & (df['subscriberCount'] <= high)]

    df = predict_gap(model, df)
    filtered = df[(df['gap'] > df['gap'].quantile(0.9)) & (df['gap_ratio'] >= 2)]

    # 텍스트 병합
    texts = filtered['title'].apply(clean_text).tolist()

    # 키워드 분석
    tfidf_result = tfidf_keywords(texts)
    bert_result = bertopic_keywords(texts)
    intersection_result = calculate_intersection(tfidf_result, bert_result)

    results = {
        'outliers': filtered[['videoID', 'channelID', 'title', 'gap', 'uploadDate', 'subscriberCount', 'channelTitle']].to_dict(orient='records'),
        'tfidf_keywords': tfidf_result,
        'bertopic_keywords': bert_result,
        'intersection_keywords': intersection_result
    }
    return results

# ------------------- 저장 -------------------
def save_results(results, filename="underdog_results.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"저장 완료: {filename}")

if __name__ == "__main__":
    results = analyze()
    save_results(results)
