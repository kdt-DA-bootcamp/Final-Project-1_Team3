import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import platform

# 한글 폰트 설정 (Windows와 Mac에 따라 설정 다름)
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
elif platform.system() == "Darwin":  # macOS
    font_path = "/Library/Fonts/AppleGothic.ttf"
else:
    font_path = None

# 결과 불러오기
@st.cache_data
def load_results():
    try:
        with open("underdog_results.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"파일을 불러오는 중 오류 발생: {e}")
        return {}

results = load_results()

st.title("반짝 영상 키워드 분석")
st.markdown("""
대형 채널이 아님에도 예상 조회수를 뛰어넘은 반짝 영상들을 분석하여 키워드를 추출했습니다.  
많이 등장한 키워드를 **워드 클라우드**로 시각화하여 한눈에 확인할 수 있습니다.
""")

# --- BERTopic 키워드 표시 ---
if 'bertopic_keywords' in results:
    bert_df = pd.DataFrame(results['bertopic_keywords'])
    if not bert_df.empty and 'keyword' in bert_df.columns and 'score' in bert_df.columns:
        bert_df['score'] = (bert_df['score'] / bert_df['score'].max() * 100).round(2)  # 정규화
        top_bert = bert_df.sort_values(by='score', ascending=False).head(15)
        bert_keywords = ", ".join(top_bert['keyword'])

        with st.expander("키워드 점수표"):
            st.dataframe(bert_df[['keyword', 'score']].sort_values(by='score', ascending=False)[:10])

        # --- 워드 클라우드 ---
        word_freq = dict(zip(bert_df['keyword'], bert_df['score']))

        # 워드 클라우드 생성
        wordcloud = WordCloud(
            font_path=font_path,  # 한글 폰트 지정
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
        st.warning("BERTopic 키워드 분석 결과는 있지만 컬럼이 누락되었습니다.")
else:
    st.warning("BERTopic 키워드 분석 결과가 포함되어 있지 않습니다.")

# --- 이상치 영상 목록 ---
st.markdown("---")
st.subheader("예측보다 높은 조회수를 기록한 영상들")

if 'outliers' in results:
    outlier_df = pd.DataFrame(results['outliers'])
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
        st.info("이상치 영상이 없습니다.")
else:
    st.error("분석된 이상치 영상 데이터가 없습니다.")