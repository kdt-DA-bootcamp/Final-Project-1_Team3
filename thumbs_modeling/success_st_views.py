import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS, to_rgb
import platform

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
    try:
        rgb = to_rgb(hex_code)
        closest_name = min(CSS4_COLORS, key=lambda name: sum((a - b) ** 2 for a, b in zip(to_rgb(CSS4_COLORS[name]), rgb)))
        return closest_name.capitalize()
    except:
        return hex_code

# --- 유사 색상 그룹화 함수 ---
def group_color_name(color_name):
    color_name = color_name.lower()

    # 유사 색상 그룹화
    common_groups = {
        "gray": ["gray", "slategray", "dimgray", "darkgray", "lightgray", "silver", "gainsboro"],
        "black": ["black", "darkslategray", "verydarkgray"],
        "white": ["white", "whitesmoke", "ghostwhite", "floralwhite", "aliceblue"],
        "brown": ["brown", "saddlebrown", "rosybrown", "burlywood", "maroon"],
        "blue": ["blue", "navy", "dodgerblue", "skyblue", "lightblue", "steelblue", "powderblue", "darkcyan"],
        "green": ["green", "lime", "darkgreen", "olivedrab", "seagreen", "lightgreen", "palegreen", "greenyellow", "darkolivegreen"],
        "red": ["red", "crimson", "orangered", "darkred", "indianred", "lightcoral"],
        "yellow": ["yellow", "gold", "lightyellow", "khaki", "goldenrod", "lemonchiffon", "wheat"],
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

# --- 블랙/화이트 및 그레이 계열 필터링 함수 ---
def is_black_or_gray_or_white(color_name):
    color_name = color_name.lower()
    return any(kw in color_name for kw in [
        # 블랙 계열
        "black", "gray", "darkgray", "dimgray", "slategray", "silver", "gainsboro",
        # 화이트 계열
        "white", "whitesmoke", "ghostwhite", "floralwhite", "aliceblue"
    ])

# --- 데이터 로드 함수 ---
@st.cache_data
def load_color_data():
    try:
        color_analysis_df = pd.read_csv("color_analysis_views.csv")

        def parse_color_distribution(color_str):
            try:
                if pd.isna(color_str) or color_str == '':
                    return {}
                # RGB 튜플을 문자열로 변환
                color_str = re.sub(r'\((\d+), (\d+), (\d+)\)', r'"\1, \2, \3"', color_str)
                color_dict = json.loads(color_str.replace("'", "\""))
                parsed_data = {tuple(map(int, k.split(", "))): v for k, v in color_dict.items()}
                return parsed_data
            except Exception as e:
                st.error(f"색상 데이터 파싱 오류: {e}")
                return {}

        color_analysis_df['high_view_colors'] = color_analysis_df['high_view_colors'].apply(parse_color_distribution)
        color_analysis_df['low_view_colors'] = color_analysis_df['low_view_colors'].apply(parse_color_distribution)

        return color_analysis_df
    except Exception as e:
        st.error(f"파일을 불러오는 중 오류 발생: {e}")
        return pd.DataFrame()

# --- 시각화 함수 ---
def visualize_color_pie(ax, color_distribution, title):
    grouped_colors = {}
    for rgb, ratio in color_distribution.items():
        hex_code = '#%02x%02x%02x' % rgb
        color_name = get_color_name(hex_code)
        # 블랙/화이트/그레이 계열 필터링
        if is_black_or_gray_or_white(color_name):
            continue
        grouped_name = group_color_name(color_name)
        grouped_colors[grouped_name] = grouped_colors.get(grouped_name, 0) + ratio

    sorted_colors = sorted(grouped_colors.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_colors:
        ax.text(0.5, 0.5, "유효 색상 없음", ha='center', va='center', fontsize=14)
        return

    color_names, ratios = zip(*sorted_colors)
    pie_colors = [CSS4_COLORS.get(name.lower(), "gray") for name in color_names]

    ax.pie(ratios, labels=color_names, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)

# --- Streamlit 인터페이스 ---
st.title("카테고리별 썸네일 색상 분석")
st.markdown("""
썸네일의 컬러톤 분포를 조회수 기준으로 비교해보았습니다.
썸네일을 돋보이게 할 포인트 색상 위주로 참고하시면 좋습니다.
""")

category_map = {
    1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
    6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
}

color_analysis_df = load_color_data()
selected_category = st.selectbox("카테고리 선택", list(category_map.keys()), format_func=lambda x: category_map[x])

# --- 레이아웃 설정 (좌우로 나란히) ---
col1, col2 = st.columns(2)

with col1:
    st.header("상위 조회수 컬러톤")
    fig, ax = plt.subplots(figsize=(4, 4))
    filtered_df = color_analysis_df[color_analysis_df['categoryID'] == selected_category]
    if not filtered_df.empty:
        high_view_colors = filtered_df.iloc[0]['high_view_colors']
        visualize_color_pie(ax, high_view_colors, "상위 조회수")
        st.pyplot(fig)

with col2:
    st.header("하위 조회수 컬러톤")
    fig, ax = plt.subplots(figsize=(4, 4))
    if not filtered_df.empty:
        low_view_colors = filtered_df.iloc[0]['low_view_colors']
        visualize_color_pie(ax, low_view_colors, "하위 조회수")
        st.pyplot(fig)