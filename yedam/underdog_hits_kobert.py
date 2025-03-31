import os
import ast
import pickle
import nltk
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
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

# ------------------- 텍스트 합치기 -------------------
def combine_text(df):
    def merge(row):
        tags = ', '.join(row['tags']) if isinstance(row['tags'], list) else str(row['tags'])
        return f"{row['title']} {tags}"
    return df.assign(text=df.apply(merge, axis=1))

# ------------------- 임베딩 & 유사도 -------------------
def compute_embeddings(df, model):
    texts = df['text'].tolist()
    return model.encode(texts, convert_to_tensor=True, device='cuda')

def analyze_keywords(outlier_df, normal_df, model, top_n=20):
    outlier_df = combine_text(outlier_df)
    normal_df = combine_text(normal_df)
    emb_out = compute_embeddings(outlier_df, model)
    emb_norm = compute_embeddings(normal_df, model)

    scores = util.cos_sim(emb_out, emb_norm)
    keyword_scores = {}
    for i in range(len(outlier_df)):
        top_indices = torch.topk(scores[i], k=min(top_n, len(normal_df))).indices
        for j in top_indices:
            j = j.item()
            text = normal_df.iloc[j]['text']
            for word in text.split():
                word = word.replace("##", "")  # BERT subword 제거
                if word.isalnum() and len(word) > 1:
                    keyword_scores[word] = keyword_scores.get(word, 0) + float(scores[i][j])

    sorted_kw = sorted(keyword_scores.items(), key=lambda x: -x[1])
    return pd.DataFrame([{'keyword': k, 'score': round(v, 4)} for k, v in sorted_kw[:top_n]])

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

    # GPU 설정된 SentenceTransformer
    kobert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device='cuda')
    keywords_df = analyze_keywords(filtered, df.drop(filtered.index), kobert)

    results = {
        'outliers': filtered[['videoID', 'channelID', 'title', 'gap', 'uploadDate', 'subscriberCount', 'channelTitle']].to_dict(orient='records'),
        'keywords_diff': keywords_df.to_dict(orient='records')
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