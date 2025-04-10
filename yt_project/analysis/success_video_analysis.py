import sys
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from config.db_config import get_db_engine

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def convert_duration_to_int(duration_str):
    try:
        if duration_str.isdigit():
            return int(duration_str)
        else:
            import isodate
            duration = isodate.parse_duration(duration_str)
            return int(duration.total_seconds())
    except Exception:
        return 0

def load_data():
    try:
        engine = get_db_engine()
        if engine is None:
            print("DB 엔진 생성 실패")
            return pd.DataFrame()
        
        # SQL 쿼리로 데이터 가져오기
        videos_df = pd.read_sql("SELECT * FROM video", engine)
        channels_df = pd.read_sql("SELECT * FROM channel", engine)
        print("데이터 불러오기 성공")

        # 데이터 컬럼 확인
        print("Videos Data Columns:", videos_df.columns)
        print("Channels Data Columns:", channels_df.columns)

        # 데이터 타입 변환
        videos_df['duration'] = videos_df['duration'].apply(convert_duration_to_int)

        # 병합 시 필요한 컬럼이 있는지 확인
        if 'channelID' not in videos_df.columns or 'channelID' not in channels_df.columns:
            print("데이터 컬럼명이 올바르지 않습니다.")
            return pd.DataFrame()

        # 데이터 병합
        merged_df = pd.merge(videos_df, channels_df, on="channelID", how="left")
        print("병합된 데이터 컬럼 확인:", merged_df.columns)

        # 분석에 필요한 컬럼 확인
        required_columns = ['subscriberCount', 'videosCount', 'duration', 'likeCount', 
                            'commentCount', 'categoryID', 'viewCount']
        for col in required_columns:
            if col not in merged_df.columns:
                print(f"필수 컬럼 {col}이(가) 없습니다.")
                return pd.DataFrame()

        return merged_df
    except Exception as e:
        print(f"데이터 불러오기 실패: {e}")
        return pd.DataFrame()

def train_model(df):
    features = ['subscriberCount', 'videosCount', 'duration', 'likeCount', 'commentCount', 'categoryID']
    df = df[features + ['viewCount']].dropna()
    X = df[features]
    y = np.log1p(df['viewCount'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("모델 학습 완료")
    return model

def predict_gap(model, df):
    df = df.copy()
    df['videosCount'] = df.groupby('channelID')['videoID'].transform('count')
    features = ['subscriberCount', 'videosCount', 'duration', 'likeCount', 'commentCount', 'categoryID']
    X = df[features].fillna(0)
    df['predicted_viewCount'] = np.expm1(model.predict(X))
    df['gap'] = df['viewCount'] - df['predicted_viewCount']
    df['gap_ratio'] = df['viewCount'] / (df['predicted_viewCount'] + 1)
    print("Gap 계산 완료")
    return df

def clean_text(text):
    return str(text).replace("#", "").replace(",", "").replace(".", "").lower()

def tfidf_keywords(texts):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).flatten()
    keywords = sorted(zip(feature_names, scores), key=lambda x: -x[1])
    return [{'keyword': k, 'score': round(v, 4)} for k, v in keywords[:20]]

def bertopic_keywords(texts):
    model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device='cuda')
    topic_model = BERTopic(embedding_model=model, language="multilingual", verbose=True)
    topics, _ = topic_model.fit_transform(texts)
    keywords = []
    for topic in sorted(set(topics)):
        if topic != -1:
            words = topic_model.get_topic(topic)
            keywords.extend([{'keyword': word, 'score': round(weight * 100, 2)} for word, weight in words])
    return keywords

def analyze():
    df = load_data()
    if df.empty:
        print("분석할 데이터가 없습니다.")
        return {}

    model = train_model(df)
    low = df['subscriberCount'].quantile(0.3)
    high = df['subscriberCount'].quantile(0.7)
    df = df[(df['subscriberCount'] > low) & (df['subscriberCount'] <= high)]
    print(f"중형 채널 필터링 후 데이터 개수: {len(df)}")
    df = predict_gap(model, df)
    results = {}
    for category_id in sorted(df['categoryID'].unique()):
        category_df = df[df['categoryID'] == category_id]
        print(f"카테고리 {category_id} 분석 중: 데이터 수 {len(category_df)}")
        filtered = category_df[(category_df['gap'] > category_df['gap'].quantile(0.9)) & (category_df['gap_ratio'] >= 1.5)]
        if filtered.empty:
            print(f"카테고리 {category_id}: 이상치 없음")
            continue
        texts = filtered['title'].apply(clean_text).tolist()
        tfidf_result = tfidf_keywords(texts)
        bert_result = bertopic_keywords(texts)

        # Timestamp → 문자열 변환
        outlier_df = filtered[['videoID', 'channelID', 'title', 'gap', 'uploadDate', 'subscriberCount', 'channelTitle']].copy()
        outlier_df['uploadDate'] = pd.to_datetime(outlier_df['uploadDate']).dt.strftime("%Y-%m-%d")
        outliers = outlier_df.to_dict(orient='records')

        results[category_id] = {
            'outliers': outliers,
            'tfidf_keywords': tfidf_result,
            'bertopic_keywords': bert_result
        }
    return results

def save_results_csv(results, filename="data/underdog_results_all.csv"):
    try:
        records = []
        for category, data in results.items():
            record = {
                "categoryID": category,
                "bertopic_keywords": json.dumps(data.get("bertopic_keywords", []), ensure_ascii=False),
                "tfidf_keywords": json.dumps(data.get("tfidf_keywords", []), ensure_ascii=False),
                "outliers": json.dumps(data.get("outliers", []), ensure_ascii=False)
            }
            records.append(record)
        if records:
            final_df = pd.DataFrame(records)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            final_df.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"모든 카테고리 데이터를 하나로 CSV 저장 완료: {filename}")
        else:
            print("저장할 데이터가 없습니다.")
    except Exception as e:
        print(f"CSV 저장 실패: {e}")

if __name__ == "__main__":
    results = analyze()
    save_results_csv(results)