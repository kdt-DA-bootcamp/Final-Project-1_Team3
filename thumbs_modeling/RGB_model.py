import os
import pandas as pd
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine
import re

load_dotenv()

# ------------------- DB 불러오기 -------------------
def load_video_thumbnail_data():
    try:
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT')
        database = os.getenv('DB_NAME')
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

        # 비디오와 썸네일 데이터 불러오기 (조회수 추가)
        video_df = pd.read_sql("SELECT videoID, categoryID, viewCount FROM video", engine)
        thumbnail_df = pd.read_sql("SELECT thumbnailURL, colorpalette FROM thumbnail", engine)

        # 썸네일 URL에서 videoID 추출
        thumbnail_df['videoID'] = thumbnail_df['thumbnailURL'].apply(
            lambda x: re.search(r'vi/([^/]+)/hqdefault\.jpg', x).group(1) if re.search(r'vi/([^/]+)/hqdefault\.jpg', x) else None
        )

        # 병합하여 비디오 ID와 카테고리 ID, 컬러 정보 함께 가져오기
        merged_df = pd.merge(video_df, thumbnail_df, on="videoID", how="inner")
        print("DB에서 데이터 로드 완료.")
        return merged_df
    except Exception as e:
        print(f"DB에서 데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

# 컬러 데이터 파싱 함수
def parse_colorpalette(colorpalette):
    try:
        # JSON 형식 처리
        if isinstance(colorpalette, str):
            colors = json.loads(colorpalette.replace("'", "\""))
        elif isinstance(colorpalette, list):
            colors = colorpalette
        else:
            return []

        parsed_colors = []
        for color in colors:
            rgb = tuple(color['rgb'])
            ratio = color['ratio']
            parsed_colors.append((rgb, ratio))
        return parsed_colors
    except:
        return []

# 색상 비율 집계 함수
def aggregate_color_data(group):
    color_count = {}
    for colors in group['parsed_colors']:
        for rgb, ratio in colors:
            color_count[rgb] = color_count.get(rgb, 0) + ratio
    # 정규화하여 비율 합이 1이 되도록 처리
    total_ratio = sum(color_count.values())
    if total_ratio > 0:
        color_count = {rgb: ratio / total_ratio for rgb, ratio in color_count.items()}
    return color_count

# 조회수 상/하위 색상 분석 함수
def analyze_color_by_view(data):
    color_analysis = []

    for category_id, group in data.groupby('categoryID'):
        # 조회수 기준 상위 10%, 하위 10%
        high_threshold = group['viewCount'].quantile(0.9)
        low_threshold = group['viewCount'].quantile(0.1)

        high_views = group[group['viewCount'] >= high_threshold]
        low_views = group[group['viewCount'] <= low_threshold]

        # 상위 조회수 색상 비율 집계
        high_color_dist = aggregate_color_data(high_views)

        # 하위 조회수 색상 비율 집계
        low_color_dist = aggregate_color_data(low_views)

        color_analysis.append({
            'categoryID': category_id,
            'high_view_colors': high_color_dist,
            'low_view_colors': low_color_dist
        })

    return pd.DataFrame(color_analysis)

# 데이터 저장 함수
def save_color_data(df, filename):
    try:
        # 컬러 데이터를 JSON 문자열로 변환하여 저장
        df['high_view_colors'] = df['high_view_colors'].apply(lambda x: json.dumps(x, ensure_ascii=False))
        df['low_view_colors'] = df['low_view_colors'].apply(lambda x: json.dumps(x, ensure_ascii=False))
        df.to_csv(filename, index=False)
        print(f"색상 분석 데이터를 '{filename}'로 저장했습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

# 메인 실행 함수
def main():
    # 데이터 불러오기
    merged_data = load_video_thumbnail_data()

    if merged_data.empty:
        print("데이터가 없습니다. 프로그램을 종료합니다.")
        return

    # 컬러 데이터를 파싱하여 새로운 컬럼으로 추가
    merged_data['parsed_colors'] = merged_data['colorpalette'].apply(parse_colorpalette)

    # 조회수 기반 색상 분석
    color_analysis_df = analyze_color_by_view(merged_data)

    # 데이터 저장
    save_color_data(color_analysis_df, "color_analysis_views.csv")

    # 결과 확인
    print(color_analysis_df.head())

if __name__ == "__main__":
    main()