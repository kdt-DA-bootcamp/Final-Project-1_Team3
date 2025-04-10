import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud
from keyword_visualizer import load_keyword_trends, visualize_trend_keywords, show_top_keywords
from thumbnail_visualizer_full import (
    load_color_data, visualize_color_comparison, visualize_top3_detailed_colors, deserialize_colors,
    display_thumbnail_recommendations  # 추가된 함수 임포트
)
from success_video_visualizer import load_success_videos, visualize_wordcloud, show_success_videos
from font import set_korean_font, get_font_path
from thumbnail_visualizer_full import (
    load_color_data,
    visualize_color_comparison,
    visualize_top3_detailed_colors,
    deserialize_colors
)

# 한글 폰트 설정
set_korean_font()

# Streamlit 앱 메인 함수
def main():
    # 페이지 기본 설정
    st.set_page_config(layout="wide")
    with st.sidebar:
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.image("../data/youtube_logo.png", width=30)
        with col2:
            st.markdown("""
                <h2 style='margin:0; padding-top:5px; display: inline-block; vertical-align: middle;'>CreatorLens</h2>
            """, unsafe_allow_html=True)

            menu = st.radio("메뉴", ["트렌드 분석", "스마트 추천", "반짝 영상 분석"])

    if menu == "트렌드 분석":
        show_trend_analysis()

    elif menu == "스마트 추천":
        show_smart_recommendation()

    elif menu == "반짝 영상 분석":
        show_high_impact_videos()

# 트렌드 분석 페이지
def show_trend_analysis():
    st.title("트렌드 리포트")
    results = load_keyword_trends()
    if results:
        categories = list(results.keys())
        selected_category = st.selectbox("카테고리 선택", categories)
        if selected_category in results:
            category_data = pd.DataFrame(results[selected_category])
            top_keywords = show_top_keywords(category_data)
            selected_keyword = st.selectbox("키워드 선택", top_keywords['keywords'].unique())
            visualize_trend_keywords(category_data, selected_keyword)

# 스마트 추천 페이지
def show_smart_recommendation():
    st.title("스마트 추천")
    st.write("썸네일의 이미지 키워드, 텍스트, 컬러톤을 분석해드립니다.")
    st.write("조회수 차이에 따른 컬러톤 비율과 세부 색상 구성을 확인할 수 있습니다.")

    
    category_map = {
        1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
        6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
    }
     # ☆☆ 썸네일 추천 섹션 추가 ☆☆
     # 하나의 공통 카테고리 선택 (정수형)
    common_selected_category = st.selectbox("분석할 카테고리를 선택하세요", list(category_map.keys()), 
                                             format_func=lambda x: category_map[x])
    

    # 썸네일 추천 영역에 선택값 전달
    display_thumbnail_recommendations(common_selected_category)
    
    st.markdown("---")
    # 컬러 데이터 관련
    color_analysis_df = load_color_data()

    if color_analysis_df.empty:
        st.error("컬러 데이터를 불러오지 못했습니다.")
        return


    # selected_category = st.selectbox("카테고리 선택", list(category_map.keys()), format_func=lambda x: category_map[x])
    filtered_df = color_analysis_df[color_analysis_df['categoryID'] == common_selected_category]

    if filtered_df.empty:
        st.warning("선택한 카테고리에 대한 데이터가 없습니다.")
        return

    # 컬러 데이터 가져오기 (캐시 복원)
    high_view_colors = deserialize_colors(filtered_df.iloc[0]['high_view_colors'])
    low_view_colors = deserialize_colors(filtered_df.iloc[0]['low_view_colors'])

    # 컬러 데이터 확인
    st.subheader("상위/하위 컬러 비율 비교 (상위 10개)")
    visualize_color_comparison(high_view_colors, low_view_colors)

    view_mode = st.radio("시각화 방식 선택", ["파이차트", "색상 팔레트"])
    st.subheader("상위 컬러 Top 3 세부 색상 구성")
    mode = "pie" if view_mode == "파이차트" else "palette"
    visualize_top3_detailed_colors(high_view_colors, mode)

   

# 반짝 영상 분석 페이지
def show_high_impact_videos():
    st.title("반짝 영상 분석")
    st.markdown("""
    대형 채널이 아님에도 예상 조회수를 뛰어넘은 반짝 영상들을 카테고리별로 분석하여 키워드를 추출했습니다.  
    많이 등장한 키워드를 **워드 클라우드**로 시각화하여 한눈에 확인할 수 있습니다.
    """)

    # 카테고리 선택
    category_map = {
        1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
        6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
    }

    # 카테고리 목록 추출
    categories = list(category_map.keys())
    selected_category = st.selectbox("카테고리 선택", categories, format_func=lambda x: category_map.get(x, "알 수 없음"))

    # 결과 불러오기
    @st.cache_data
    def load_results():
        try:
            with open("../data/underdog_results.pkl", "rb") as f:
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
                font_path=get_font_path(),
                width=400, 
                height=300, 
                background_color='white', 
                colormap='viridis', 
                max_words=50
            ).generate_from_frequencies(word_freq)

            # 시각화
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig, use_container_width=False)

        else:
            st.warning(f"카테고리 '{category_map.get(selected_category, '알 수 없음')}'에 대한 키워드 분석 결과가 없습니다. 반짝 영상들 간의 키워드 공통점이 존재하지 않습니다.")
    else:
        st.warning(f"카테고리 '{category_map.get(selected_category, '알 수 없음')}'에 대한 키워드 분석 결과가 포함되어 있지 않습니다.")

    # 인기 영상 표시
    if selected_category_str in results:
        outlier_df = pd.DataFrame(results[selected_category_str]['outliers'])
        if not outlier_df.empty:
            outlier_df = outlier_df.sort_values(by='gap', ascending=False)

            st.markdown("---")
            st.subheader(f"예측보다 높은 조회수를 기록한 영상들 ({category_map[selected_category]})")

            for _, row in outlier_df.head(10).iterrows():
                upload_date = pd.to_datetime(row['uploadDate']).strftime('%Y-%m-%d')
                gap_text = f"{int(row['gap']):,}회 더 조회됨"
                subs_text = f"{int(row['subscriberCount']):,}명"
                video_id = row['videoID']
                thumbnail_url = row.get('thumbnailURL', '')

                # 썸네일 URL이 없는 경우 기본 URL 사용
                if not thumbnail_url:
                    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

                # 영상 URL 생성
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # 썸네일을 클릭하면 유튜브 영상으로 이동
                image_html = f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="300"></a>'

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