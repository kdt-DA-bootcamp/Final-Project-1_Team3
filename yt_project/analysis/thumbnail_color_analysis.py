# ------ 썸네일 컬러톤 추천 ------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import re
from config.db_config import get_db_engine
from config.db_config import get_db_engine

def extract_video_id(url):
    pattern = r'vi/([^/]+)/[^/]+\.jpg'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def load_video_thumbnail_data():
    try:
        engine = get_db_engine()
        if engine is None:
            print("DB 엔진 생성 실패")
            return pd.DataFrame()

        video_df = pd.read_sql("SELECT videoID, categoryID, viewCount FROM video", engine)
        thumbnail_df = pd.read_sql("SELECT thumbnailURL, colorpalette FROM thumbnail", engine)

        thumbnail_df['videoID'] = thumbnail_df['thumbnailURL'].apply(extract_video_id)

        if video_df.empty or thumbnail_df.empty:
            print("영상 또는 썸네일 데이터가 비어 있습니다.")
            return pd.DataFrame()

        merged_data = pd.merge(video_df, thumbnail_df, on="videoID", how="inner")
        if merged_data.empty:
            print("병합된 데이터가 없습니다.")
            return pd.DataFrame()

        print("DB에서 데이터 로드 완료.")
        return merged_data
    except Exception as e:
        print(f"DB에서 데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

def parse_colorpalette(colorpalette):
    try:
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
    except Exception:
        return []

def aggregate_color_data(group):
    color_count = {}
    for colors in group['parsed_colors']:
        for rgb, ratio in colors:
            color_count[rgb] = color_count.get(rgb, 0) + ratio

    total_ratio = sum(color_count.values())
    if total_ratio > 0:
        color_count = {rgb: ratio / total_ratio for rgb, ratio in color_count.items()}
    return color_count

def analyze_color_by_view(data):
    color_analysis = []

    for category_id, group in data.groupby('categoryID'):
        high_threshold = group['viewCount'].quantile(0.9)
        low_threshold = group['viewCount'].quantile(0.1)

        high_views = group[group['viewCount'] >= high_threshold]
        low_views = group[group['viewCount'] <= low_threshold]

        high_color_dist = aggregate_color_data(high_views)
        low_color_dist = aggregate_color_data(low_views)

        color_analysis.append({
            'categoryID': category_id,
            'high_view_colors': high_color_dist,
            'low_view_colors': low_color_dist
        })

    return pd.DataFrame(color_analysis)

def save_color_data(df, filename):
    try:
        def convert_color_dict(color_dict):
            return "{" + ", ".join([f"({k[0]}, {k[1]}, {k[2]}): {v}" for k, v in color_dict.items()]) + "}"
        
        df['high_view_colors'] = df['high_view_colors'].apply(lambda x: convert_color_dict(x))
        df['low_view_colors'] = df['low_view_colors'].apply(lambda x: convert_color_dict(x))
        
        df.to_csv(filename, index=False)
        print(f"색상 분석 데이터를 '{filename}'로 저장했습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

def main():
    merged_data = load_video_thumbnail_data()

    if merged_data.empty:
        print("데이터가 없습니다. 프로그램을 종료합니다.")
        return

    merged_data['parsed_colors'] = merged_data['colorpalette'].apply(parse_colorpalette)

    color_analysis_df = analyze_color_by_view(merged_data)

    # 데이터 저장
    save_color_data(color_analysis_df, "data/color_analysis_views.csv")

    # 결과 확인
    print("분석 결과 예시:")
    print(color_analysis_df.head())

if __name__ == "__main__":
    main()