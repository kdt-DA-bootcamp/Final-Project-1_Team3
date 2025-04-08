import pandas as pd
import streamlit as st
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy import log1p
from collections import Counter
from ast import literal_eval
from math import log1p

# 데이터 로드 함수
def load_keyword_analysis(filename="../data/keyword_analysis.csv"):
    try:
        df = pd.read_csv(filename)

        # 'uploadDate' 컬럼 확인
        if 'uploadDate' not in df.columns:
            print("🚨 [오류] 'uploadDate' 컬럼이 없습니다.")
            return pd.DataFrame()

        # 날짜 형식 변환 - datetime으로 통일하여 date()로 추출
        try:
            df['uploadDate'] = pd.to_datetime(df['uploadDate'], errors='coerce')
            print("날짜 변환 후 데이터 확인:")
            print(df[['uploadDate']].head())
            print(f"변환 후 최소 날짜: {df['uploadDate'].min()}")
            print(f"변환 후 최대 날짜: {df['uploadDate'].max()}")
            print("[디버그] uploadDate 컬럼의 데이터 타입:")
            print(df['uploadDate'].dtype)
        except Exception as e:
            print(f"[오류] 날짜 변환 실패: {e}")


        # 카테고리 매핑
        category_map = {
            1: "엔터테인먼트",
            2: "차량",
            3: "여행/음식",
            4: "게임",
            5: "스포츠",
            6: "라이프",
            7: "정치",
            8: "반려동물",
            9: "교육",
            10: "과학/기술"
        }
        df['category'] = df['categoryID'].map(category_map)
        return df
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return pd.DataFrame()

# 반짝 키워드와 꾸준히 상승 키워드 분석 함수
def analyze_keywords(df, start_date=None, end_date=None):
    results = []

    # 키워드 파싱 함수
    def parse_keywords(row):
        try:
            if isinstance(row, str):
                return [kw.strip() for kw in row.split(',') if kw.strip()]
            if isinstance(row, list):
                return row
            parsed = literal_eval(str(row))
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception:
            return []

    df = df.copy()
    df['parsed_keywords'] = df['keywords'].apply(parse_keywords)
    df['uploadDate'] = pd.to_datetime(df['uploadDate'], errors='coerce').dt.date

    # 카테고리별 분석 루프
    for cat, sub_df in df.groupby('category'):
        date_keyword_freq = {}
        for _, row in sub_df.iterrows():
            date = row['uploadDate']
            keywords = row['parsed_keywords']
            for kw in keywords:
                date_keyword_freq.setdefault(date, Counter())[kw] += 1

        keyword_freq_df = pd.DataFrame(date_keyword_freq).fillna(0).astype(int).T.sort_index()
        if keyword_freq_df.empty:
            print(f"[경고] 카테고리 '{cat}'에 대한 키워드 빈도 데이터프레임이 비어 있습니다.")
            continue

        keyword_freq_df.index = pd.to_datetime(keyword_freq_df.index).date
        all_dates = keyword_freq_df.index
        min_date = all_dates.min()
        max_date = all_dates.max()

        # 슬라이딩 윈도우 분석 시작
        for window_date in pd.date_range(start=min_date + timedelta(days=3), end=max_date).date:
            if window_date not in all_dates:
                continue

            recent_3 = keyword_freq_df.loc[window_date - timedelta(days=2): window_date]
            prev_7 = keyword_freq_df.loc[window_date - timedelta(days=9): window_date - timedelta(days=3)]

            if prev_7.empty or recent_3.empty:
                continue

            recent_mean = recent_3.mean()
            prev_mean = prev_7.mean()
            
            for kw in recent_mean.index:
                recent_val = recent_mean[kw]
                prev_val = prev_mean.get(kw, 0)
                log_score = round(log1p(recent_val) - log1p(prev_val + 1e-6), 4)
                is_new = prev_val < 1

                if start_date and window_date < start_date:
                    continue
                if end_date and window_date > end_date:
                    continue

                if log_score > 0.5:
                    results.append({
                        'category': cat,
                        'keyword': kw,
                        'type': 'flash',
                        'score': log_score if not is_new else 'NEW',
                        'uploadDate': window_date
                    })

        # 꾸준히 상승하는 키워드 (고정 방식, 전체 기간 대상)
        slopes = {}
        X = np.arange(len(keyword_freq_df)).reshape(-1, 1)
        for kw in keyword_freq_df.columns:
            y = keyword_freq_df[kw].values
            if y.sum() < 5:
                continue
            model = LinearRegression().fit(X, y)
            slopes[kw] = model.coef_[0]

        trending_sorted = sorted(slopes.items(), key=lambda x: -x[1])
        for kw, val in dict(trending_sorted[:20]).items():
            results.append({
                'category': cat,
                'keyword': kw,
                'type': 'steady',
                'score': round(val, 4),
                'uploadDate': max_date  # steady는 최신일 기준
            })

    return pd.DataFrame(results)