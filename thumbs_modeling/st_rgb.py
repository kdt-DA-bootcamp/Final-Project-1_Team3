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

# --- 블랙/화이트 및 그레이 계열 필터링 함수 ---
def is_black_or_gray_or_white(color_name):
    color_name = color_name.lower()
    return any(kw in color_name for kw in [
        # 블랙 계열
        "black", "gray", "darkgray", "dimgray", "slategray", "silver", "gainsboro",
        # 화이트 계열
        "white", "whitesmoke", "ghostwhite", "floralwhite", "aliceblue"
    ])

# --- 유사 색상 그룹화 함수 ---
def group_color_name(color_name):
    color_name = color_name.lower()
    common_groups = {
        "gray": ["gray", "slategray", "dimgray", "darkgray", "lightgray", "silver", "gainsboro"],
        "black": ["black", "darkslategray", "verydarkgray"],
        "white": ["white", "whitesmoke", "ghostwhite", "floralwhite", "aliceblue"],
        "blue": ["blue", "navy", "dodgerblue", "skyblue", "lightblue", "steelblue", "powderblue", "darkcyan"],
        "green": ["green", "lime", "darkgreen", "olivedrab", "seagreen", "lightgreen", "palegreen", "greenyellow"],
        "red": ["red", "crimson", "orangered", "darkred", "indianred", "lightcoral"],
        "yellow": ["yellow", "gold", "lightyellow", "khaki", "goldenrod", "lemonchiffon", "wheat"],
        "brown": ["brown", "saddlebrown", "rosybrown", "burlywood", "maroon"],
        "pink": ["pink", "lightpink", "hotpink", "deeppink"],
        "purple": ["purple", "violet", "plum", "orchid", "lavender", "indigo"],
        "cyan": ["cyan", "aqua", "lightcyan", "darkcyan", "paleturquoise", "lightseagreen"],
        "orange": ["orange", "darkorange", "coral", "tomato", "peachpuff", "bisque"],
    }
    for group, variations in common_groups.items():
        if any(var in color_name for var in variations):
            return group.capitalize()
    return color_name.capitalize()

# --- 데이터 로드 함수 ---
@st.cache_data
def load_color_data():
    try:
        color_analysis_df = pd.read_csv("color_analysis_views.csv")
        def parse_color_distribution(color_str):
            try:
                if pd.isna(color_str) or color_str == '':
                    return {}
                color_str = re.sub(r'\((\d+), (\d+), (\d+)\)', r'"\1, \2, \3"', color_str)
                color_dict = json.loads(color_str.replace("'", "\""))
                return {tuple(map(int, k.split(", "))): v for k, v in color_dict.items()}
            except:
                return {}
        color_analysis_df['high_view_colors'] = color_analysis_df['high_view_colors'].apply(parse_color_distribution)
        color_analysis_df['low_view_colors'] = color_analysis_df['low_view_colors'].apply(parse_color_distribution)
        return color_analysis_df
    except:
        return pd.DataFrame()

# --- 시각화 함수 1: 상위/하위 컬러 비율 비교 (상위 10개) ---
def visualize_color_comparison_top10_v2(high_colors, low_colors):
    high_grouped = {}
    low_grouped = {}

    # 상위 조회수 색상 그룹화
    for rgb, ratio in high_colors.items():
        hex_code = '#%02x%02x%02x' % rgb
        color_name = get_color_name(hex_code)
        # 블랙/그레이 필터링
        if is_black_or_gray_or_white(color_name):
            continue
        grouped_name = group_color_name(color_name)
        high_grouped[grouped_name] = high_grouped.get(grouped_name, 0) + ratio

    # 하위 조회수 색상 그룹화
    for rgb, ratio in low_colors.items():
        hex_code = '#%02x%02x%02x' % rgb
        color_name = get_color_name(hex_code)
        # 블랙/그레이 필터링
        if is_black_or_gray_or_white(color_name):
            continue
        grouped_name = group_color_name(color_name)
        low_grouped[grouped_name] = low_grouped.get(grouped_name, 0) + ratio

    # 두 색상 그룹 합치기
    combined_groups = set(high_grouped.keys()).union(set(low_grouped.keys()))

    # 상위 10개 항목 선택
    sorted_high = sorted(high_grouped.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_low = sorted(low_grouped.items(), key=lambda x: x[1], reverse=True)[:10]

    # 공통 항목 우선 포함
    top10_groups = sorted(set([item[0] for item in sorted_high] + [item[0] for item in sorted_low]), 
                          key=lambda x: (high_grouped.get(x, 0) + low_grouped.get(x, 0)), 
                          reverse=True)[:10]

    high_values = [high_grouped.get(group, 0) for group in top10_groups]
    low_values = [low_grouped.get(group, 0) for group in top10_groups]

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.4
    x = range(len(top10_groups))

    # 상위 조회수 막대 (파란색)
    ax.bar([pos - bar_width/2 for pos in x], high_values, bar_width, label="상위 조회수", color="blue", alpha=0.7)

    # 하위 조회수 막대 (주황색)
    ax.bar([pos + bar_width/2 for pos in x], low_values, bar_width, label="하위 조회수", color="orange", alpha=0.7)

    # 막대 색상 지정 (상위 파란색, 하위 주황색)
    for i, group in enumerate(top10_groups):
        ax.bar([x[i] - bar_width/2], high_values[i], bar_width, color="blue", alpha=0.7)
        ax.bar([x[i] + bar_width/2], low_values[i], bar_width, color="orange", alpha=0.7)

    # 그래프 설정
    ax.set_xticks(x)
    ax.set_xticklabels(top10_groups, rotation=45, ha='right')
    ax.set_title("상위/하위 컬러 비율 비교 (상위 10개)")
    ax.legend()

    st.pyplot(fig)

# --- 텍스트 대비 색상 결정 함수 ---
def get_contrast_text_color(bg_color):
    """배경색에 따라 흰색 또는 검정색을 반환하여 대비를 확보"""
    r, g, b = to_rgb(bg_color)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return "black" if luminance > 0.5 else "white"

# --- 세부 색상 구성 시각화 (상위 컬러 Top 3) ---
def visualize_top3_detailed_colors(high_colors, mode="pie"):
    # 상위 조회수 색상 그룹화
    high_grouped = {}
    detailed_colors = {}

    for rgb, ratio in high_colors.items():
        hex_code = '#%02x%02x%02x' % rgb
        color_name = get_color_name(hex_code)
        if is_black_or_gray_or_white(color_name):
            continue
        grouped_name = group_color_name(color_name)
        high_grouped[grouped_name] = high_grouped.get(grouped_name, 0) + ratio
        if grouped_name not in detailed_colors:
            detailed_colors[grouped_name] = {}
        detailed_colors[grouped_name][color_name] = detailed_colors[grouped_name].get(color_name, 0) + ratio

    # 상위 3개 그룹 선택
    top3_groups = sorted(high_grouped.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_names = [group[0] for group in top3_groups]

    # Streamlit 레이아웃 설정 (3개 파이차트를 한 줄에 배치)
    col1, col2, col3 = st.columns(3)

    for i, group_name in enumerate(top3_names):
        group_details = detailed_colors[group_name]

        # 그룹 내부 비율 합을 100%로 조정
        total_ratio = sum(group_details.values())
        normalized_ratios = [ratio / total_ratio for ratio in group_details.values()]

        # 비율 순으로 정렬
        sorted_details = sorted(zip(group_details.keys(), normalized_ratios), key=lambda x: x[1], reverse=True)
        color_names, ratios = zip(*sorted_details)
        hex_colors = [CSS4_COLORS.get(name.lower(), "gray") for name in color_names]

        # Streamlit 그래프와 텍스트를 같은 컬럼에 배치
        if i == 0:
            with col1:
                display_graph_and_text(group_name, color_names, ratios, hex_colors, mode)
        elif i == 1:
            with col2:
                display_graph_and_text(group_name, color_names, ratios, hex_colors, mode)
        else:
            with col3:
                display_graph_and_text(group_name, color_names, ratios, hex_colors, mode)

def display_graph_and_text(group_name, color_names, ratios, hex_colors, mode):
    if mode == "pie":
        # 파이차트 그리기
        fig, ax = plt.subplots(figsize=(4, 4))

        # 비율 필터링하여 텍스트 표시
        def autopct_func(pct):
            return f'{pct:.1f}%' if pct >= 10 else ''

        # 파이차트 라벨 필터링 수정
        filtered_labels = [name if ratio >= 0.1 else name for name, ratio in zip(color_names, ratios)]

        wedges, texts, autotexts = ax.pie(
            ratios, labels=filtered_labels, colors=hex_colors, autopct=autopct_func, startangle=90
        )

        # 텍스트 색상 조정
        for text, wedge in zip(texts, wedges):
            bg_color = wedge.get_facecolor()
            text.set_color(get_contrast_text_color(bg_color))

        ax.set_title(f"{group_name}")
        st.pyplot(fig)

    elif mode == "palette":
        # 색상 팔레트 그리기
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        x_offset = 0
        for j, (name, ratio) in enumerate(zip(color_names, ratios)):
            width = ratio * 10
            ax.add_patch(plt.Rectangle((x_offset, 0), width, 1, color=hex_colors[j]))
            # 비율이 10% 이상일 때만 텍스트 표시
            if ratio >= 0.1:
                text_color = get_contrast_text_color(hex_colors[j])
                ax.text(x_offset + width / 2, 0.5, f"{name}\n{ratio * 100:.1f}%", 
                        va="center", ha="center", fontsize=8, color=text_color)
            x_offset += width
        ax.set_xlim(0, 10)
        ax.set_title(f"{group_name}")
        st.pyplot(fig)

    # 그래프 아래에 비율 순으로 텍스트 표시
    for name, ratio in zip(color_names, ratios):
        if ratio >= 0.01:  # 비율이 1% 미만인 경우 생략
            st.markdown(f"- **{name}**: {ratio * 100:.1f}%")

# --- Streamlit 인터페이스 ---
st.title("상위 컬러 Top 3 세부 색상 구성 및 상위/하위 컬러 비교")

category_map = {
    1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
    6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
}

color_analysis_df = load_color_data()
selected_category = st.selectbox("카테고리 선택", list(category_map.keys()), format_func=lambda x: category_map[x])

filtered_df = color_analysis_df[color_analysis_df['categoryID'] == selected_category]
if not filtered_df.empty:
    high_view_colors = filtered_df.iloc[0]['high_view_colors']
    low_view_colors = filtered_df.iloc[0]['low_view_colors']

    st.header("상위/하위 컬러 비율 비교 (상위 10개)")
    visualize_color_comparison_top10_v2(high_view_colors, low_view_colors)

    view_mode = st.radio("시각화 방식 선택", ["파이차트", "색상 팔레트"])

    st.header("상위 컬러 Top 3 세부 색상 구성")
    mode = "pie" if view_mode == "파이차트" else "palette"
    visualize_top3_detailed_colors(high_view_colors, mode)