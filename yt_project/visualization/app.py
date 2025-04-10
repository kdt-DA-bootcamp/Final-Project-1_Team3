import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import altair as alt
from wordcloud import WordCloud
from keyword_visualizer import load_keyword_analysis, analyze_keywords
from thumbnail_visualizer_full import (
    load_color_data, visualize_color_comparison, visualize_top3_detailed_colors, deserialize_colors,
    display_thumbnail_recommendations
)
from success_video_visualizer import load_success_videos, visualize_wordcloud, show_success_videos
from font import set_korean_font, get_font_path
from posixpath import basename
st.set_page_config(layout="wide")

# 한글 폰트 설정
set_korean_font()

## 로고 이미지 경로 설정 함수
def get_logo_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(project_root, 'data', 'youtube_logo.png')
    return logo_path

# Streamlit 앱 메인 함수
def main():
    # 페이지 기본 설정
    with st.sidebar:
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            logo_path = get_logo_path()
            try:
                st.image(logo_path, width=30)
            except Exception as e:
                st.error(f"로고 이미지 불러오기 오류: {e}")

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
    st.markdown("카테고리와 날짜 범위를 선택해 키워드 언급이 급등하는 반짝 키워드와 언급이 안정적인 꾸준 키워드를 분석하세요.")

    df = load_keyword_analysis()
    if df.empty:
        st.error("데이터를 불러오는데 실패했습니다.")
        return

    try:
        df['uploadDate'] = pd.to_datetime(df['uploadDate'], errors='coerce').dt.date
    except Exception as e:
        st.error(f"날짜 변환 오류: {e}")
        return

    categories = sorted(df['category'].dropna().unique())
    selected_category = st.selectbox("카테고리를 선택하세요", categories)
    category_df = df[df['category'] == selected_category]

    min_date = df['uploadDate'].min()
    max_date = df['uploadDate'].max()

    date_range = st.date_input(
        "분석할 날짜 범위를 선택하세요",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.error("날짜 범위를 올바르게 선택해 주세요.")
        return

    st.write(f"선택한 날짜 범위: {date_range[0]} ~ {date_range[1]}")

    result_df = analyze_keywords(category_df, start_date=date_range[0], end_date=date_range[1])
    if result_df.empty:
        st.error("분석 결과 데이터가 비어 있습니다.")
        return

    total_keywords = len(result_df)
    new_keywords = result_df[result_df['score'].astype(str).str.upper() == 'NEW']
    st.success(f"전체 분석 키워드 수: {total_keywords}개 | 신규 키워드 수: {len(new_keywords)}개")

    top_n = st.selectbox("몇 개의 상위 키워드를 볼까요?", [10, 20, 30, 50], index=1)
    tabs = st.tabs(["반짝 키워드", "꾸준 키워드"])

    # 반짝 키워드
    with tabs[0]:
        st.subheader(f"{selected_category} - 반짝 키워드 (TOP {top_n})")
        flash_df = result_df[result_df['type'] == 'flash'].copy()
        flash_df['numeric_score'] = pd.to_numeric(flash_df['score'], errors='coerce')
        flash_df['is_new'] = flash_df['score'].astype(str).str.strip().str.upper() == 'NEW'
        flash_df = flash_df.sort_values(['is_new', 'numeric_score'], ascending=[False, False]).head(top_n)

        flash_display = flash_df[['keyword', 'score', 'is_new']].rename(columns={
            'keyword': '키워드',
            'score': '급등 비율 또는 NEW',
            'is_new': '신규 키워드 여부'
        })

        flash_display.reset_index(drop=True, inplace=True)
        flash_display.index = flash_display.index + 1
        flash_display.index.name = '순위'

        st.dataframe(flash_display, use_container_width=True)
        st.download_button("반짝 키워드 CSV 다운로드", flash_df.to_csv(index=False), "flash_keywords.csv", "text/csv")

        st.subheader('키워드 상세 보기')
        for i, (idx, row) in enumerate(flash_df.iterrows()):
            keyword = row['keyword']
            mask = category_df['keywords'].astype(str).str.contains(keyword, case=False, na=False)
            related_videos = category_df[mask]

            top_video = related_videos.sort_values('viewCount', ascending=False).head(1)

            with st.expander(f"{keyword}"):
                st.write(f"- 점수: {row['score']}")
                st.write(f"- 신규 여부: {'신규 키워드' if row['is_new'] else '기존 키워드'}")
                st.write("→ 이 키워드는 최근 특정 이슈 또는 트렌드에 의해 급격히 떠오르고 있습니다.")
                if not top_video.empty:
                    video_row = top_video.iloc[0]
                    video_url = f"https://www.youtube.com/watch?v={video_row['videoID']}"
                    view_count = f"{int(video_row['viewCount']):,}"
                    subscriber_count = f"{int(video_row['subscriberCount']):,}" if pd.notna(video_row['subscriberCount']) else "정보 없음"
                    st.markdown(f"""
                    **대표 영상 추천**: [{video_row['title']}]({video_url})
                    조회수: {view_count}회  
                    구독자 수: {subscriber_count}명
                    """)
                    
                    # 추세선 그래프
                    trend_range = st.radio("추세선 기간 선택", ["전체", "선택한 기간"], horizontal=True, key=f"steady_trend_{keyword}_{i}")
                    if trend_range == "선택한 기간":
                        filtered_videos = related_videos[
                            (related_videos['uploadDate'] >= date_range[0]) &
                            (related_videos['uploadDate'] <= date_range[1])
                        ]
                    else:
                        filtered_videos = related_videos


                    time_trend = filtered_videos.groupby('uploadDate').size().reset_index(name='count')
                    if not time_trend.empty:
                        line_chart = alt.Chart(time_trend).mark_line(point=True).encode(
                            x=alt.X('uploadDate:T', title='업로드일'),
                            y=alt.Y('count:Q', title='언급 수'), 
                            tooltip=[
                                alt.Tooltip('uploadDate:T', title='업로드일'), 
                                alt.Tooltip('count:Q', title='언급 수')
                                ]
                        ).properties(title=f"{keyword} 언급량 추이 ({trend_range})")
                        st.altair_chart(line_chart, use_container_width=True)
                    else:
                        st.info("선택한 기간 내에 해당 키워드가 언급된 영상이 없습니다.")
                else:
                    st.markdown("대표 영상을 찾을 수 없습니다.")
    
    # 꾸준 키워드
    with tabs[1]:
        st.subheader(f"{selected_category} - 꾸준 키워드 (TOP {top_n})")
        steady_df = result_df[result_df['type'] == 'steady'].copy()
        steady_df['numeric_score'] = pd.to_numeric(steady_df['score'], errors='coerce')
        steady_df = steady_df.sort_values("numeric_score", ascending=False).head(top_n)

        if not steady_df.empty:
            chart = alt.Chart(steady_df).mark_bar(size=20).encode(
                x=alt.X("numeric_score:Q", title="기울기"),
                y=alt.Y("keyword:N", sort='-x', title="키워드"),
                tooltip=["keyword", "score"]
            ).properties(height=500)
            st.altair_chart(chart, use_container_width=True)

            st.subheader('키워드 상세 보기')
            for i, (idx, row) in enumerate(steady_df.iterrows()):
                keyword = row['keyword']
                with st.expander(f"{keyword}"):
                    st.write(f"- 점수 (기울기): {row['score']}")
                    mask = category_df['keywords'].astype(str).str.contains(keyword, case=False, na=False)
                    related_videos = category_df[mask]
                    top_video = related_videos.sort_values('viewCount', ascending=False).head(1)
                    if not top_video.empty:
                        video_row = top_video.iloc[0]
                        video_url = f"https://www.youtube.com/watch?v={video_row['videoID']}"
                        view_count = f"{int(video_row['viewCount']):,}"
                        subscriber_count = f"{int(video_row['subscriberCount']):,}" if pd.notna(video_row['subscriberCount']) else "정보 없음"
                        st.markdown(f"""
                        **대표 영상**: [{video_row['title']}]({video_url})  
                        조회수: {view_count}회  
                        구독자 수: {subscriber_count}명
                        """)
                    else:
                        st.markdown("대표 영상을 찾을 수 없습니다.")

                    # 추세선 그래프
                    trend_range = st.radio("추세선 기간 선택", ["전체", "선택한 기간"], horizontal=True, key=f"flash_trend_{keyword}_{i}")
                    if trend_range == "선택한 기간":
                        filtered_videos = related_videos[
                            (related_videos['uploadDate'] >= date_range[0]) &
                            (related_videos['uploadDate'] <= date_range[1])
                        ]
                    else:
                        filtered_videos = related_videos

                    time_trend = filtered_videos.groupby('uploadDate').size().reset_index(name='count')
                    if not time_trend.empty:
                        line_chart = alt.Chart(time_trend).mark_line(point=True).encode(
                            x='uploadDate:T',
                            y='count:Q',
                            tooltip=['uploadDate:T', 'count']
                        ).properties(title=f"{keyword} 언급량 추이 ({trend_range})")
                        st.altair_chart(line_chart, use_container_width=True)
                    else:
                        st.info("선택한 기간 내에 해당 키워드가 언급된 영상이 없습니다.")

# 스마트 추천 페이지
def show_smart_recommendation():
    st.title("스마트 추천")

    category_map = {
        1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
        6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
    }
    
    tabs = st.tabs(["썸네일 추천", "업로드 시간 분석"])

    with tabs[0]:
        st.subheader("썸네일 추천")
        st.write("조회수 차이에 따른 컬러톤 비율과 세부 색상 구성을 확인할 수 있습니다.")
        selected_category = st.selectbox("분석할 카테고리를 선택하세요", list(category_map.keys()), format_func=lambda x: category_map[x])
        display_thumbnail_recommendations(selected_category)

        st.markdown("---")
        color_analysis_df = load_color_data()
        if color_analysis_df.empty:
            st.error("컬러 데이터를 불러오지 못했습니다.")
            return

        filtered_df = color_analysis_df[color_analysis_df['categoryID'] == selected_category]
        if filtered_df.empty:
            st.warning("선택한 카테고리에 대한 데이터가 없습니다.")
            return

        high_view_colors = deserialize_colors(filtered_df.iloc[0]['high_view_colors'])
        low_view_colors = deserialize_colors(filtered_df.iloc[0]['low_view_colors'])

        st.subheader("상위/하위 컬러 비율 비교 (상위 10개)")
        visualize_color_comparison(high_view_colors, low_view_colors)

        st.subheader("상위 컬러 Top 3 세부 색상 구성")
        mode = "palette"
        visualize_top3_detailed_colors(high_view_colors, mode)

    with tabs[1]:
        st.subheader("카테고리별 업로드 시간대 분석")
        st.write("카테고리별로 동영상이 업로드된 시간대 분포를 확인할 수 있습니다.")

        df = load_keyword_analysis()
        if df.empty or 'uploadDate' not in df.columns or 'categoryID' not in df.columns:
            st.error("업로드 시간 분석을 위한 데이터가 없습니다.")
            return
        
        # 카테고리ID 숫자형 변환
        df['categoryID'] = pd.to_numeric(df['categoryID'], errors='coerce')
        df = df.dropna(subset=['categoryID'])
        df['categoryID'] = df['categoryID'].astype(int)

        # 날짜 파싱만 공통으로 적용
        df['uploadDate'] = pd.to_datetime(df['uploadDate'], errors='coerce')
        df = df.dropna(subset=['uploadDate'])

        # 카테고리 선택
        category_map = {
            1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
            6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
        }
        selected_category_id = st.selectbox("카테고리를 선택하세요", list(category_map.keys()), format_func=lambda x: category_map[x])

        # 카테고리별 필터링 후 시간/요일 추출
        cat_df = df[df['categoryID'] == selected_category_id].copy()
        if cat_df.empty:
            st.warning("선택한 카테고리에 해당하는 데이터가 없습니다.")
            return

        # 시간 및 요일 추출 및 0시 보정
        cat_df['hour'] = cat_df['uploadDate'].dt.hour
        cat_df['weekday'] = cat_df['uploadDate'].dt.day_name()

        is_midnight = cat_df['hour'] == 0
        cat_df.loc[is_midnight, 'hour'] = 24
        cat_df.loc[is_midnight, 'weekday'] = (cat_df.loc[is_midnight, 'uploadDate'] - pd.Timedelta(days=1)).dt.day_name()

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        cat_df['weekday'] = pd.Categorical(cat_df['weekday'], categories=weekday_order, ordered=True)

        # 바 차트
        hour_counts = cat_df['hour'].value_counts().sort_index()
        hour_df = pd.DataFrame({'hour': hour_counts.index, 'count': hour_counts.values})

        hour_order = list(range(1, 25))
        hour_labels = [f"{i}시" for i in hour_order]

        hour_df['hour_label'] = pd.Categorical(
            hour_df['hour'].astype(str) + "시",
            categories=hour_labels,
            ordered=True
        )

        bar_chart = alt.Chart(hour_df).mark_bar().encode(
            x=alt.X('hour:O', title='업로드 시간대 (시)', sort=list(range(1, 25))),
            y=alt.Y('count:Q', title='업로드된 영상 수'),
            tooltip=[alt.Tooltip('hour:O', title='업로드 시각'), 'count']
        ).properties(
            title=f"'{category_map[selected_category_id]}' 카테고리 시간대별 업로드 분포",
            height=300
        )

        # 히트맵
        heatmap_df = cat_df.groupby(['weekday', 'hour']).size().reset_index(name='count')
        heatmap_df['hour_label'] = heatmap_df['hour'].astype(str) + "시"
        heatmap_df['weekday'] = pd.Categorical(heatmap_df['weekday'], categories=weekday_order, ordered=True)

        heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
            x=alt.X('hour_label:O', title='업로드 시간대 (시 단위)', sort=hour_labels),
            y=alt.Y('weekday:O', title='요일'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['weekday', 'hour_label', 'count']
        ).properties(
            title=f"'{category_map[selected_category_id]}' 카테고리 요일-시간대별 업로드 히트맵",
            height=300
        )

        # 출력
        st.altair_chart(bar_chart, use_container_width=True)
        st.markdown("### 요일 + 시간대별 업로드 히트맵")
        st.altair_chart(heatmap_chart, use_container_width=True)


# 반짝 영상 분석 페이지
@st.cache_data
def load_success_videos():
    try:
        df = pd.read_csv("./data/underdog_results_all.csv", encoding="utf-8-sig")
        if 'categoryID' not in df.columns:
            st.error("CSV 파일에 'categoryID' 컬럼이 없습니다.")
            return {}
        results = {}
        for category, group in df.groupby('categoryID'):
            results[str(category)] = {
                'bertopic_keywords': group.iloc[0]['bertopic_keywords'],
                'outliers': group.iloc[0]['outliers']
            }
        return results
    except Exception as e:
        st.error(f"파일을 불러오는 중 오류 발생: {e}")
        return {}

def show_high_impact_videos():
    st.title("반짝 영상 분석")
    st.markdown("""
    대형 채널이 아님에도 예상 조회수를 뛰어넘은 반짝 영상들을 카테고리별로 분석하여 키워드를 추출했습니다.
    많이 등장한 키워드를 **워드 클라우드**로 시각화하여 한눈에 확인할 수 있습니다.
    """)
    category_map = {
        1: '엔터테인먼트', 2: '차량', 3: '여행&음식', 4: '게임', 5: '스포츠',
        6: '라이프', 7: '정치', 8: '반려동물', 9: '교육', 10: '과학/기술'
    }
    categories = list(category_map.keys())
    selected_category = st.selectbox("카테고리 선택", categories,
                                     format_func=lambda x: category_map.get(x, "알 수 없음"))
    selected_category_str = str(selected_category)
    results = load_success_videos()
    # 키워드 표시
    if selected_category_str in results:
        keyword_str = results[selected_category_str]['bertopic_keywords']
        try:
            keyword_data = ast.literal_eval(keyword_str)
        except Exception as e:
            st.error(f"키워드 데이터를 파싱하는 중 오류 발생: {e}")
            return
        bert_df = pd.DataFrame(keyword_data)
        if not bert_df.empty and 'keyword' in bert_df.columns and 'score' in bert_df.columns:
            bert_df['score'] = (bert_df['score'] / bert_df['score'].max() * 100).round(2)
            top_bert = bert_df.sort_values(by='score', ascending=False).head(15)
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
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig, use_container_width=False)
        else:
            st.warning(f"카테고리 '{category_map.get(selected_category, '알 수 없음')}'에 대한 키워드 분석 결과가 없습니다. 반짝 영상들 간의 키워드 공통점이 존재하지 않습니다.")
    else:
        st.warning(f"카테고리 '{category_map.get(selected_category, '알 수 없음')}'에 대한 키워드 분석 결과가 포함되어 있지 않습니다.")
    # 인기 영상 표시 (이상치 데이터 처리; 필요 시 파싱 작업 포함)
    if selected_category_str in results:
        outlier_str = results[selected_category_str]['outliers']
        try:
            outlier_data = ast.literal_eval(outlier_str)
        except Exception as e:
            st.error(f"이상치 데이터를 파싱하는 중 오류 발생: {e}")
            return
        outlier_df = pd.DataFrame(outlier_data)
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
                if not thumbnail_url:
                    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                image_html = f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="300"></a>'
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