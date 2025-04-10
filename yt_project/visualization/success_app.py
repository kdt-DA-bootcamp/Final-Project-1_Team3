import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import streamlit as st
import pandas as pd
import pickle
import json
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import platform
from font import get_font_path, set_korean_font
from matplotlib.colors import CSS4_COLORS, to_rgb
from config.db_config import get_db_engine

# ------ 기본 설정 ------
def main():
    # 페이지 기본 설정
    st.set_page_config(layout="wide")
    with st.sidebar:
        col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.image("data/youtube_logo.png", width=30)
    with col2:
        st.markdown("<h2 style='margin:0; padding-top:5px;'>CreatorLens</h2>", unsafe_allow_html=True)
        # 메뉴 표시
        menu = st.radio(
            "메뉴",
            ["트렌드분석", "스마트추천", "반짝영상분석"]
        )

    # 선택된 메뉴에 따라 다른 페이지(화면) 표시
    if menu == "트렌드분석":
        show_trend_analysis()
    elif menu == "스마트추천":
        show_smart_recommendation()
    else:
        show_high_impact_shorts()

# 한글폰트 설정
set_korean_font()
get_font_path()

# db 연결
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

# ------ 트렌드 분석 ------
def show_trend_analysis():
    # 상단 타이틀
    st.title("트렌드 리포트")
    st.write("카테고리를 선택하여 꾸준히 상승하는 키워드와, 반짝 상승한 키워드를 확인하세요.")
    
    # 데이터 로드 함수
    @st.cache_data
    def load_results(filename="data/keyword_trends.pkl"):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"파일을 불러오는 중 오류 발생: {e}")
            return {}

    results = load_results()

    # 스트림릿 UI
    st.subheader("꾸준히 상승 중인 키워드 분석")
    st.markdown("""
    최근 데이터 분석을 통해 **꾸준히 상승하는 키워드**를 추출했습니다.  
    카테고리를 선택하여 주요 키워드와 그 추세를 확인해보세요.
    """)

    category_map = {
        '1': '엔터테인먼트', '2': '차량', '3': '여행', '4': '게임', '5': '스포츠',
        '6': '라이프', '7': '정치', '8': '반려동물', '9': '교육/Howto', '10': '과학/기술'
    }

    categories = list(results.keys())
    selected_category = st.selectbox("카테고리 선택", categories, format_func=lambda x: category_map.get(x, "알 수 없음"))

    # 데이터 확인
    if selected_category in results:
        category_data = pd.DataFrame(results[selected_category])
        if not category_data.empty:
            st.subheader(f"카테고리: {category_map.get(selected_category, '알 수 없음')}")

            # 성장 점수 상위 10개 키워드
            top_keywords = category_data.sort_values(by="final_score", ascending=False).head(10)

            st.markdown("### 꾸준히 상승 중인 키워드 Top 10")
            st.dataframe(top_keywords[['keywords', 'final_score']])

            # 키워드 선택
            selected_keyword = st.selectbox("상승 추세 확인할 키워드 선택", top_keywords['keywords'].unique())

            # 선택한 키워드의 추세 데이터 필터링
            keyword_data = category_data[category_data['keywords'] == selected_keyword]

            # 상승 추세 시각화
            if not keyword_data.empty:
                st.markdown(f"### 키워드 '{selected_keyword}'의 주별 등장 추세")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(keyword_data['period'], keyword_data['count'], marker='o', linestyle='-', label='키워드 등장 빈도')
                ax.set_title(f"'{selected_keyword}' 키워드의 주별 등장 추세")
                ax.set_xlabel("주차")
                ax.set_ylabel("등장 빈도")
                ax.grid(True)
                plt.xticks(keyword_data['period'].unique())  # 주차 표시
                plt.legend()

                # 그래프 출력
                st.pyplot(fig)
            else:
                st.warning(f"키워드 '{selected_keyword}'에 대한 데이터가 없습니다.")
        else:
            st.warning(f"카테고리 '{category_map.get(selected_category, '알 수 없음')}'에 대한 데이터가 없습니다.")
    else:
        st.error(f"선택한 카테고리에 대한 분석 결과가 없습니다.")


# ------ 스마트 추천 ------
def show_smart_recommendation():
    st.title("스마트추천")
    st.write("썸네일의 이미지 키워드, 텍스트, 컬러톤을 분석해드립니다.")

    # 색상 이름 변환 함수
    def get_color_name(hex_code):
        try:
            rgb = to_rgb(hex_code)
            closest_name = min(CSS4_COLORS, key=lambda name: sum((a - b) ** 2 for a, b in zip(to_rgb(CSS4_COLORS[name]), rgb)))
            return closest_name.capitalize()
        except:
            return hex_code

    # 블랙/화이트 및 그레이 계열 필터링 함수
    def is_black_or_gray_or_white(color_name):
        color_name = color_name.lower()
        return any(kw in color_name for kw in [
            # 블랙 계열
            "black", "gray", "darkgray", "dimgray", "slategray", "silver", "gainsboro",
            # 화이트 계열
            "white", "whitesmoke", "ghostwhite", "floralwhite", "aliceblue"
        ])

    # 유사 색상 그룹화 함수
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

    # 데이터 로드 함수
    @st.cache_data
    def load_color_data():
        try:
            color_analysis_df = pd.read_csv("data/color_analysis_views.csv")
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

    # 시각화 함수 1: 상위/하위 컬러 비율 비교 (상위 10개)
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

    # Streamlit 인터페이스 ---
    st.subheader("상위 컬러 Top 3 세부 색상 구성 및 상위/하위 컬러 비교")

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

        st.subheader("상위/하위 컬러 비율 비교 (상위 10개)")
        visualize_color_comparison_top10_v2(high_view_colors, low_view_colors)

        view_mode = st.radio("시각화 방식 선택", ["파이차트", "색상 팔레트"])

        st.subheader("상위 컬러 Top 3 세부 색상 구성")
        mode = "pie" if view_mode == "파이차트" else "palette"
        visualize_top3_detailed_colors(high_view_colors, mode)

# ------ 작지만 강한 영상 ------
def show_high_impact_shorts():
    st.title("반짝영상분석")
    st.markdown("""
    대형 채널이 아님에도 예상 조회수를 뛰어넘은 반짝 영상들을 카테고리별로 분석하여 키워드를 추출했습니다.  
    많이 등장한 키워드를 **워드 클라우드**로 시각화하여 한눈에 확인할 수 있습니다.
    """)

    # 카테고리 선택
    category_map = {
        1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
        6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
    }
    
    categories = list(category_map.keys())
    selected_category = st.selectbox("카테고리 선택", categories, format_func=lambda x: category_map[x])

    # 결과 불러오기
    @st.cache_data
    def load_results():
        try:
            with open("data/underdog_results.pkl", "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"파일을 불러오는 중 오류 발생: {e}")
            return {}

    results = load_results()

    # 키워드 표시
    selected_category_str = str(selected_category)
    if selected_category_str in results:
        bert_df = pd.DataFrame(results[selected_category_str]['bertopic_keywords'])
        if not bert_df.empty and 'keyword' in bert_df.columns and 'score' in bert_df.columns:
            bert_df['score'] = (bert_df['score'] / bert_df['score'].max() * 100).round(2) 
            top_bert = bert_df.sort_values(by='score', ascending=False).head(15)
            bert_keywords = ", ".join(top_bert['keyword'])

            with st.expander("키워드 점수표"):
                st.dataframe(bert_df[['keyword', 'score']].sort_values(by='score', ascending=False)[:10])

            word_freq = dict(zip(bert_df['keyword'], bert_df['score']))

            wordcloud = WordCloud(
                font_path=get_font_path(),  # 한글 폰트 경로 지정
                width=800, 
                height=400, 
                background_color='white', 
                colormap='viridis', 
                max_words=50
            ).generate_from_frequencies(word_freq)

            # 시각화
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        else:
            st.warning(f"카테고리 '{category_map[selected_category]}'에 대한 키워드 분석 결과는 있지만 컬럼이 누락되었습니다.")
    else:
        st.warning(f"카테고리 '{category_map[selected_category]}'에 대한 키워드 분석 결과가 포함되어 있지 않습니다.")


    # 이상치 영상 목록
    st.markdown("---")
    st.subheader(f"예측보다 높은 조회수를 기록한 영상들 ({category_map[selected_category]})")

    if selected_category_str in results:
        outlier_df = pd.DataFrame(results[selected_category_str]['outliers'])
        if not outlier_df.empty:
            outlier_df = outlier_df.sort_values(by='gap', ascending=False)

            for _, row in outlier_df.head(10).iterrows():
                upload_date = pd.to_datetime(row['uploadDate']).strftime('%Y-%m-%d')
                gap_text = f"{int(row['gap']):,}회 더 조회됨"
                subs_text = f"{int(row['subscriberCount']):,}명"
                video_id = row['videoID']
                thumbnail_url = row.get('thumbnailURL', '')

                # 썸네일 URL이 DB에 없는 경우 기본 URL로 대체
                if not thumbnail_url:
                    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

                # 영상 URL 생성
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # 썸네일을 클릭하면 유튜브 영상으로 이동
                if thumbnail_url:
                    image_html = f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="300"></a>'
                else:
                    image_html = "썸네일 없음"

                # Streamlit에 표시
                st.markdown(f"""
                - **{row['title']}**  
                예측 대비 **{gap_text}**, 업로드일: {upload_date}, 구독자 수: {subs_text}, 채널명: {row['channelTitle']}
                """)
                st.markdown(image_html, unsafe_allow_html=True)
        else:
            st.info(f"카테고리 '{category_map[selected_category]}'에 대한 이상치 영상이 없습니다.")
    else:
        st.error(f"카테고리 '{category_map[selected_category]}'에 대한 분석된 이상치 영상 데이터가 없습니다.")


if __name__ == "__main__":
    main()