import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine
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

# ------------------- 텍스트 전처리 -------------------
def clean_text(text):
    text = str(text).replace("#", "").replace(",", "").replace(".", "").lower()
    return text

def combine_text(row):
    tags = ', '.join(row['tags']) if isinstance(row['tags'], list) else str(row['tags'])
    return f"{row['title']} {tags}"

# ------------------- BERTopic 키워드 추출 -------------------
def extract_keywords(text_series):
    # TF-IDF 기반으로 키워드 추출
    vectorizer = TfidfVectorizer(max_features=20)  # 상위 20개 키워드 추출
    tfidf_matrix = vectorizer.fit_transform(text_series)
    keywords = vectorizer.get_feature_names_out()
    return keywords


# ------------------- 키워드 상승 추세 분석 -------------------
def analyze_trend(df):
    # 데이터 복사본 만들기
    df = df.copy()

    # 날짜 형식 변환
    df['uploadDate'] = pd.to_datetime(df['uploadDate'])

    # 주 단위로 나누기 (7일 단위)
    df['period'] = df['uploadDate'].dt.isocalendar().week

    # 제목과 태그 결합하여 키워드 추출
    df['combined_text'] = df.apply(lambda row: combine_text(row), axis=1)

    # TF-IDF로 키워드 추출
    keywords = extract_keywords(df['combined_text'])

    # 각 문장에 대해 상위 키워드를 리스트로 저장
    def get_keywords(text):
        return [kw for kw in keywords if kw in text]

    # 키워드 리스트화 (빈 경우 빈 리스트 할당)
    df['keywords'] = df['combined_text'].apply(lambda x: get_keywords(x) if isinstance(x, str) else [])

    # explode를 사용하기 위해 빈 리스트를 빈 문자열로 대체
    df['keywords'] = df['keywords'].apply(lambda x: x if len(x) > 0 else [''])

    # 키워드 등장 빈도 계산
    trend_df = df.explode('keywords').groupby(['categoryID', 'period', 'keywords']).size().reset_index(name='count')

    # 키워드의 주기적 등장 여부 확인 (4주 연속 등장 여부)
    trend_df['consistency'] = trend_df.groupby(['categoryID', 'keywords'])['count'].transform(lambda x: x.rolling(4).count().fillna(0))

    # 상승 추세 계산: 4주 이동 평균으로 등장 빈도 증가 여부 확인
    trend_df['growth'] = trend_df.groupby(['categoryID', 'keywords'])['count'].pct_change().fillna(0)
    trend_df['growth_score'] = trend_df['growth'].rolling(window=4).mean().fillna(0)

    # 성장 점수와 꾸준함 점수를 합산하여 최종 점수 계산
    trend_df['final_score'] = trend_df['growth_score'] * trend_df['consistency']

    # 상승 키워드 상위 20개
    top_keywords = trend_df.groupby(['categoryID', 'keywords'])['final_score'].max().reset_index()
    top_keywords = top_keywords.sort_values(by='final_score', ascending=False).head(20)

    return trend_df[trend_df['keywords'].isin(top_keywords['keywords'])][['categoryID', 'period', 'keywords', 'count', 'final_score']]

# ------------------- 전체 분석 -------------------
def analyze():
    df = load_data()

    # 결과를 저장할 딕셔너리
    results = {}

    # 카테고리별로 분석 수행
    for category in df['categoryID'].unique():
        print(f"Analyzing category {category}")

        # 카테고리별 데이터 필터링
        category_df = df[df['categoryID'] == category].copy()

        # 키워드 상승 분석 수행
        keyword_df = analyze_trend(category_df)

        # 결과가 제대로 데이터프레임인지 확인
        if isinstance(keyword_df, pd.DataFrame) and not keyword_df.empty:
            results[str(category)] = keyword_df.to_dict(orient='records')
            print(f"분석 완료: 카테고리 {category}, 데이터 개수: {len(keyword_df)}")
        else:
            print(f"분석 실패: 카테고리 {category}, 데이터가 없습니다.")

    # 결과 저장
    save_results(results, filename="keyword_trends.pkl")
    print("분석 결과 저장 완료: keyword_trends.pkl")
    return results

# ------------------- 저장 -------------------
def save_results(results, filename="keyword_trends.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"분석 결과 저장 완료: {filename}")

if __name__ == "__main__":
    results = analyze()