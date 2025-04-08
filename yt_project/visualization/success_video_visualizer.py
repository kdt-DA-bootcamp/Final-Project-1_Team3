import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from font import set_korean_font, get_font_path

# 한글 폰트 설정
set_korean_font()

# 데이터 로드 함수
def load_success_videos(filename="../data/underdog_results.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"파일을 불러오는 중 오류 발생: {e}")
        return {}

# 반짝 영상 워드클라우드 시각화 함수
def visualize_wordcloud(bert_df):
    if 'keyword' not in bert_df.columns or 'score' not in bert_df.columns:
        st.warning("키워드 분석 데이터가 올바르지 않습니다.")
        st.write("데이터 컬럼 확인:", bert_df.columns.tolist())
        return

    word_freq = dict(zip(bert_df['keyword'], bert_df['score']))
    wordcloud = WordCloud(
        font_path=get_font_path(),
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# 반짝 영상 목록 표시 함수
def show_success_videos(videos_df):
    if videos_df.empty:
        st.warning("반짝 영상 데이터가 없습니다.")
        return

    # 데이터 구조 확인
    for _, row in videos_df.iterrows():
        # 안전하게 데이터 접근
        title = row.get('title', '제목 없음')
        gap = row.get('gap', '조회수 정보 없음')
        upload_date = row.get('uploadDate', '업로드 날짜 없음')
        channel_title = row.get('channelTitle', '채널명 없음')
        subscriber_count = row.get('subscriberCount', '구독자 수 없음')
        video_id = row.get('videoID', None)

        # 썸네일 URL이 없는 경우 videoID를 이용하여 URL 생성
        thumbnail_url = row.get('thumbnailURL', f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg") if video_id else 'https://via.placeholder.com/300'

        # 조회수 차이 표시
        gap_text = f"{int(gap):,}회 더 조회됨" if isinstance(gap, (int, float)) else "조회수 정보 없음"
        subs_text = f"{int(subscriber_count):,}명" if isinstance(subscriber_count, (int, float)) else "구독자 수 정보 없음"

        # 영상 정보 표시
        st.markdown(f"""
        - **{title}**  
        예측 대비 **{gap_text}**, 업로드일: {upload_date}, 구독자 수: {subs_text}, 채널명: {channel_title}
        """)
        st.image(thumbnail_url, width=300)
