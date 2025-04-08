import pandas as pd
import json
import re
import ast
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import CSS4_COLORS, to_rgb
import pickle

# 색상 이름 변환 함수
def get_color_name(hex_code):
    try:
        rgb = to_rgb(hex_code)
        closest_name = min(CSS4_COLORS, key=lambda name: sum((a - b) ** 2 for a, b in zip(to_rgb(CSS4_COLORS[name]), rgb)))
        return closest_name.capitalize()
    except:
        return hex_code
    
# RGB 튜플을 HEX 문자열로 변환하는 함수
def rgb_to_hex(rgb):
    try:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    except Exception as e:
        st.error(f"HEX 변환 오류: {e}")
        return '#000000'

# 블랙/화이트 및 그레이 계열 필터링 함수
def is_black_or_gray_or_white(color_name):
    color_name = color_name.lower()
    return any(kw in color_name for kw in [
        "black", "gray", "darkgray", "dimgray", "slategray", "silver", "gainsboro",
        "white", "whitesmoke", "ghostwhite", "floralwhite", "aliceblue"
    ])

def get_contrast_text_color(bg_color):
    rgb = to_rgb(bg_color)
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    return "black" if luminance > 0.5 else "white"

# 유사 색상 그룹화 함수
def group_color_name(color_name):
    color_name = color_name.lower()
    common_groups = {
        'gray': ['gray', 'slategray', 'dimgray', 'darkgray', 'lightgray', 'silver', 'gainsboro'],
        'black': ['black', 'darkslategray', 'verydarkgray'],
        'white': ['white', 'whitesmoke', 'ghostwhite', 'floralwhite', 'aliceblue'],
        'blue': ['blue', 'navy', 'dodgerblue', 'skyblue', 'lightblue', 'steelblue', 'powderblue', 'darkcyan'],
        'green': ['green', 'lime', 'darkgreen', 'olivedrab', 'seagreen', 'lightgreen', 'palegreen', 'greenyellow'],
        'red': ['red', 'crimson', 'orangered', 'darkred', 'indianred', 'lightcoral'],
        'yellow': ['yellow', 'gold', 'lightyellow', 'khaki', 'goldenrod', 'lemonchiffon', 'wheat'],
        'brown': ['brown', 'saddlebrown', 'rosybrown', 'burlywood', 'maroon'],
        'pink': ['pink', 'lightpink', 'hotpink', 'deeppink'],
        'purple': ['purple', 'violet', 'plum', 'orchid', 'lavender', 'indigo'],
        'cyan': ['cyan', 'aqua', 'lightcyan', 'darkcyan', 'paleturquoise', 'lightseagreen'],
        'orange': ['orange', 'darkorange', 'coral', 'tomato', 'peachpuff', 'bisque']
    }
    for group, variations in common_groups.items():
        if any(var in color_name for var in variations):
            return group.capitalize()
    return color_name.capitalize()

# 문자열 컬러 데이터를 딕셔너리로 변환하는 함수
def parse_color_distribution(color_str):
    try:
        if pd.isna(color_str) or color_str == '':
            return {}

        # 안전한 파싱 (파이썬 딕셔너리 형식 처리)
        try:
            color_dict = ast.literal_eval(color_str)
        except (SyntaxError, ValueError):
            try:
                # JSON 형식일 경우
                color_dict = json.loads(color_str)
            except json.JSONDecodeError:
                st.error("컬러 데이터 파싱 실패")
                return {}

        # 키를 문자열로 변환하여 캐싱 가능하게 처리
        return {str(k): v for k, v in color_dict.items()}
    except Exception as e:
        st.error(f"컬러 데이터 파싱 오류: {e}")
        return {}

# 캐시에서 복구할 때 키를 튜플로 변환하는 함수
def deserialize_colors(color_dict):
    try:
        def convert_key(k):
            # 문자열 키를 튜플로 변환
            if k.startswith("(") and k.endswith(")"):
                return tuple(map(int, k.strip("()").split(", ")))
            return k

        return {convert_key(k): v for k, v in color_dict.items()}
    except Exception as e:
        st.error(f"컬러 데이터 복구 오류: {e}")
        return {}

# 데이터 로드 함수
@st.cache_data
def load_color_data():
    try:
        color_analysis_df = pd.read_csv("../data/color_analysis_views.csv")

        # 컬러 데이터를 파싱하여 딕셔너리로 변환
        color_analysis_df['high_view_colors'] = color_analysis_df['high_view_colors'].apply(parse_color_distribution)
        color_analysis_df['low_view_colors'] = color_analysis_df['low_view_colors'].apply(parse_color_distribution)

        return color_analysis_df
    except Exception as e:
        st.error(f"컬러 데이터 로드 오류: {e}")
        return pd.DataFrame()

# 시각화 함수: 상위/하위 컬러 비교 (상위 10개)
def visualize_color_comparison(high_colors, low_colors):
    high_grouped = {}
    low_grouped = {}

    # 상위 조회수 색상 그룹화
    for rgb, ratio in high_colors.items():
        # RGB 튜플을 HEX로 변환하여 사용
        if isinstance(rgb, tuple):
            hex_code = rgb_to_hex(rgb)
        else:
            hex_code = rgb

        # 색상 이름으로 변환
        color_name = get_color_name(hex_code)

        # 블랙/그레이 필터링
        if is_black_or_gray_or_white(color_name):
            continue

        # 유사 색상 그룹화
        grouped_name = group_color_name(color_name)
        high_grouped[grouped_name] = high_grouped.get(grouped_name, 0) + ratio

    # 하위 조회수 색상 그룹화
    for rgb, ratio in low_colors.items():
        # RGB 튜플을 HEX로 변환하여 사용
        if isinstance(rgb, tuple):
            hex_code = rgb_to_hex(rgb)
        else:
            hex_code = rgb

        # 색상 이름으로 변환
        color_name = get_color_name(hex_code)

        # 블랙/그레이 필터링
        if is_black_or_gray_or_white(color_name):
            continue

        # 유사 색상 그룹화
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
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    bar_width = 0.4
    x = range(len(top10_groups))

    # 상위 조회수 막대 (파란색)
    ax.bar([pos - bar_width/2 for pos in x], high_values, bar_width, label="상위 조회수", color="blue", alpha=0.7)

    # 하위 조회수 막대 (주황색)
    ax.bar([pos + bar_width/2 for pos in x], low_values, bar_width, label="하위 조회수", color="orange", alpha=0.7)

    # 그래프 설정
    ax.set_xticks(x)
    ax.set_xticklabels(top10_groups, rotation=45, ha='right')
    ax.set_title("상위/하위 컬러 비율 비교 (상위 10개)")
    ax.legend()

    st.pyplot(fig, use_container_width=False)

# 상위 컬러 Top 3 세부 색상 구성 시각화
def visualize_top3_detailed_colors(high_colors, mode="pie"):
    high_grouped = {}
    detailed_colors = {}

    # 색상 그룹화
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

# 그래프와 텍스트를 동시에 표시하는 함수
def display_graph_and_text(group_name, color_names, ratios, hex_colors, mode):
    if mode == "pie":
        # 파이차트 그리기
        fig, ax = plt.subplots(figsize=(4, 4))

        # 파이차트 라벨 필터링
        def autopct_func(pct):
            return f'{pct:.1f}%' if pct >= 10 else ''

        wedges, texts, autotexts = ax.pie(
            ratios, labels=color_names, colors=hex_colors, autopct=autopct_func, startangle=90
        )

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
        if ratio >= 0.01:
            st.markdown(f"- **{name}**: {ratio * 100:.1f}%")




# ☆☆ 아래부터 썸네일 추천 기능 코드 ☆☆

def load_data(imc_path: str, refined_path: str):
    """
    pickle 파일들을 로드하여 이미지 키워드 추천 데이터와 썸네일 문구 추천 데이터를 반환합니다.
    """
    with open(imc_path, 'rb') as f:
        df_IMC = pickle.load(f)
    with open(refined_path, 'rb') as f:
        df_refined = pickle.load(f)
    return df_IMC, df_refined

def get_thumbnail_keywords(df_IMC, group_number: int, top_n: int = 20):
    if group_number in df_IMC:
        return df_IMC[group_number][:top_n]  # (키워드, point) 튜플 리스트
    else:
        return []

def get_thumbnail_texts(df_refined, group_number: int, top_n: int = 10):
    """
    df_refined에서 'category' 컬럼이 group_number와 일치하는 행의 'refined_text'를 상위 top_n개 추출하여 반환합니다.
    'category' 컬럼은 문자열 형태로 저장되어 있다고 가정합니다.
    """
    df_category = df_refined[df_refined['category'] == str(group_number)]
    if not df_category.empty:
        return df_category['refined_text'].tolist()[:top_n]
    else:
        return []


def display_thumbnail_recommendations(selected_category=None):
    """
    Streamlit에서 썸네일 추천 섹션을 표 + 그래프로 나란히 표시합니다.
    """
    st.header("썸네일 추천")

    # 데이터 경로
    imc_path = '../data/translated_category_recommendations.pkl'
    refined_path = '../data/refined_recommendations.pkl'

    # 데이터 로드
    df_IMC, df_refined = load_data(imc_path, refined_path)
    group_number = selected_category

    # 썸네일 이미지 키워드 추천
    st.subheader("썸네일 이미지 키워드 추천")
    keywords_data = get_thumbnail_keywords(df_IMC, group_number, top_n=20)

    if keywords_data:
        # 키워드 + point 여부 확인
        if isinstance(keywords_data[0], tuple):
            df_keywords = pd.DataFrame(keywords_data, columns=["키워드", "point"])
        else:
            df_keywords = pd.DataFrame(keywords_data, columns=["키워드"])
        df_keywords["점수"] = (df_keywords["point"] * 100).round(1).astype(str)
        # 순위 추가
        df_keywords.index = df_keywords.index + 1
        df_keywords.reset_index(inplace=True)
        df_keywords.rename(columns={"index": "순위"}, inplace=True)

        # 좌우 2컬럼으로 분할: 왼쪽은 표, 오른쪽은 그래프
        col1, col2 = st.columns([1, 1.5])  # 오른쪽 비율 살짝 넓게

        with col1:
            st.dataframe(df_keywords[["순위", "키워드", "점수"]], hide_index=True, use_container_width=True)

        with col2:
            fig = plot_keyword_bar_chart(df_keywords)
            st.pyplot(fig)

    else:
        st.warning("해당 그룹의 이미지 추천 데이터가 없습니다.")


    # 썸네일 문구 추천 출력
    st.subheader("썸네일 문구 추천")
    texts = get_thumbnail_texts(df_refined, group_number, top_n=10)
    if texts:
        df_texts = pd.DataFrame({"문구": texts})
        df_texts.index = df_texts.index + 1
        df_texts.reset_index(inplace=True)
        df_texts.rename(columns={"index": "순위"}, inplace=True)
        st.dataframe(df_texts, hide_index=True)
    else:
        st.write("해당 그룹의 문구 추천 데이터가 없습니다.")

    



from font import get_font_path

def plot_keyword_bar_chart(df_keywords):
    df_plot = df_keywords.sort_values("순위", ascending=False)
    n = len(df_plot)

    # 그래프 세로 크기를 키워드 수에 따라 동적으로 조정
    height_per_item = 0.15  # 한 항목당 0.4인치 높이
    fig_height = max(4, n * height_per_item)  # 최소 높이 4인치

    fig, ax = plt.subplots(figsize=(6, fig_height))
    ax.barh(df_plot["키워드"], df_plot["point"], color="skyblue")
    ax.set_xlabel("point")

    for i, (score, label) in enumerate(zip(df_plot["point"], df_plot["키워드"])):
        ax.text(score + 0.005, i, f"{score:.2f}", va='center')

    plt.tight_layout()
    return fig
