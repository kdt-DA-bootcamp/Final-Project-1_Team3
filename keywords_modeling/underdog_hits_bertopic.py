import os
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

# ------------------- 전체 분석 -------------------
def analyze():
    df = load_data()
    model = train_model(df)

    # 중형 채널 필터링
    low = df['subscriberCount'].quantile(0.3)
    high = df['subscriberCount'].quantile(0.7)
    df = df[(df['subscriberCount'] > low) & (df['subscriberCount'] <= high)]

    print(f"중형 채널 필터링 후 데이터 개수: {len(df)}")

    df = predict_gap(model, df)
    
    # categoryID 타입 변환 및 확인
    if 'categoryID' in df.columns:
        df['categoryID'] = df['categoryID'].astype(int)
        print(f"카테고리 ID 유형: {df['categoryID'].dtype}")
        print(f"카테고리 ID 고유값: {df['categoryID'].unique()}")

    results = {}

    # 카테고리별 분석
    for category_id in sorted(df['categoryID'].unique()):
        # categoryID 필터링
        category_df = df[df['categoryID'] == category_id]
        
        # 중형 채널 필터링 후 카테고리별 데이터 개수 확인
        print(f"카테고리 {category_id}: 데이터 개수 {len(category_df)}")

        # 카테고리별 데이터가 있는지 확인
        if category_df.empty:
            print(f"카테고리 {category_id}: 데이터 없음")
            continue
        
        # 이상치 필터링 조건 완화: gap_ratio를 2에서 1.5로 조정
        filtered = category_df[(category_df['gap'] > category_df['gap'].quantile(0.9)) & (category_df['gap_ratio'] >= 1.5)]

        # 이상치 데이터가 없는 경우 로그로 표시
        if filtered.empty:
            print(f"카테고리 {category_id}: 이상치 없음")
            continue
        
        print(f"카테고리 {category_id}: 이상치 {len(filtered)}개")

        texts = filtered['title'].apply(clean_text).tolist()
        tfidf_result = tfidf_keywords(texts)
        bert_result = bertopic_keywords(texts)

        results[category_id] = {
            'outliers': filtered[['videoID', 'channelID', 'title', 'gap', 'uploadDate', 'subscriberCount', 'channelTitle']].to_dict(orient='records'),
            'tfidf_keywords': tfidf_result,
            'bertopic_keywords': bert_result
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
