import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS, to_hex, to_rgb
import platform
from PIL import ImageColor

# --- 한글 폰트 설정 ---
def set_korean_font():
    if platform.system() == "Windows":
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == "Darwin":  # macOS
        plt.rc('font', family='AppleGothic')
    else:  # Linux (Ubuntu 등)
        plt.rc('font', family='NanumGothic')

set_korean_font()

# --- 색상 이름 변환 함수 ---
def get_color_name(hex_code):
    """HEX 코드로부터 가장 가까운 색상 이름을 반환"""
    try:
        rgb = to_rgb(hex_code)
        closest_name = min(CSS4_COLORS, key=lambda name: sum((a - b) ** 2 for a, b in zip(to_rgb(CSS4_COLORS[name]), rgb)))
        return closest_name.capitalize()
    except:
        return hex_code

# --- 색상 그룹화 함수 ---
def group_color_name(color_name):
    """비슷한 색상명을 그룹화"""
    color_name = color_name.lower()

    # 공통 계열로 묶기
    common_groups = {
        "gray": ["gray", "slategray", "dimgray", "darkgray", "lightgray", "silver", "gainsboro"],
        "black": ["black", "darkslategray", "verydarkgray"],
        "white": ["white", "whitesmoke", "ghostwhite", "floralwhite", "aliceblue"],
        "blue": ["blue", "navy", "dodgerblue", "skyblue", "lightblue", "steelblue"],
        "green": ["green", "lime", "darkgreen", "olivedrab", "seagreen", "lightgreen", "palegreen", "greenyellow"],
        "red": ["red", "crimson", "orangered", "darkred", "indianred"],
        "yellow": ["yellow", "gold", "lightyellow", "khaki", "goldenrod", "lemonchiffon"],
        "brown": ["brown", "maroon", "saddlebrown", "chocolate", "rosybrown", "burlywood"],
        "pink": ["pink", "lightpink", "hotpink", "deeppink"],
        "purple": ["purple", "violet", "plum", "orchid", "lavender", "indigo"],
        "cyan": ["cyan", "aqua", "lightcyan", "darkcyan", "paleturquoise", "lightseagreen"],
        "orange": ["orange", "darkorange", "coral", "tomato", "peachpuff", "bisque"],
    }

    for group, variations in common_groups.items():
        if any(var in color_name for var in variations):
            return group.capitalize()

    # 기본 색상으로 사용
    return color_name.capitalize()

# --- 데이터 로드 함수 ---
@st.cache_data
def load_color_data():
    try:
        color_analysis_df = pd.read_csv("color_analysis.csv")
        def parse_color_distribution(color_str):
            try:
                color_str = re.sub(r'\((\d+), (\d+), (\d+)\)', r'"\1, \2, \3"', color_str)
                color_dict = json.loads(color_str.replace("'", "\""))
                return {tuple(map(int, k.split(", "))): v for k, v in color_dict.items()}
            except:
                return {}
        color_analysis_df['color_distribution'] = color_analysis_df['color_distribution'].apply(parse_color_distribution)
        return color_analysis_df
    except Exception as e:
        st.error(f"파일을 불러오는 중 오류 발생: {e}")
        return pd.DataFrame()

# --- 필터링 함수 ---
def is_black_or_gray(color_name):
    color_name = color_name.lower()
    return any(kw in color_name for kw in ["black", "gray", "darkgray", "dimgray", "slategray", "silver", "gainsboro"])

# --- 시각화 함수 ---
def visualize_color_pie(category_id):
    plt.figure(figsize=(8, 8))
    filtered_df = color_analysis_df[color_analysis_df['categoryID'] == category_id]
    if filtered_df.empty:
        st.warning(f"카테고리 '{category_map[category_id]}'에 대한 색상 데이터가 없습니다.")
        return

    row = filtered_df.iloc[0]
    color_distribution = row['color_distribution']
    filtered_colors = {rgb: ratio for rgb, ratio in color_distribution.items()
                       if not is_black_or_gray(get_color_name('#%02x%02x%02x' % rgb))}
    
    grouped_colors = {}
    for rgb, ratio in filtered_colors.items():
        hex_code = '#%02x%02x%02x' % rgb
        color_name = get_color_name(hex_code)
        grouped_name = group_color_name(color_name)
        grouped_colors[grouped_name] = grouped_colors.get(grouped_name, 0) + ratio

    sorted_colors = sorted(grouped_colors.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_colors:
        st.warning(f"카테고리 '{category_map[category_id]}'에 유효 색상 데이터가 없습니다.")
        return

    color_names, ratios = zip(*sorted_colors)
    pie_colors = [CSS4_COLORS.get(name.lower(), "gray") for name in color_names]
    plt.pie(ratios, labels=color_names, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    plt.title(f"카테고리 '{category_map[category_id]}' 썸네일 색상 분포")
    st.pyplot(plt)

# --- Streamlit 인터페이스 ---
st.title("카테고리별 썸네일 색상 분석")
category_map = {
    1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
    6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
}
color_analysis_df = load_color_data()
selected_category = st.selectbox("카테고리 선택", list(category_map.keys()), format_func=lambda x: category_map[x])
visualize_color_pie(selected_category)