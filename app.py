import streamlit as st
import pandas as pd
import altair as alt
import datetime

# 페이지 설정
st.set_page_config(page_title="키워드 트렌드 분석", layout="wide")
st.title("📊 카테고리별 키워드 트렌드 분석")
st.markdown("카테고리와 날짜 범위를 선택해 반짝 키워드와 꾸준히 상승하는 키워드를 분석하세요.")

# CSV 불러오기
df = pd.read_csv("keyword_trend_by_category.csv")
df['uploadDate'] = pd.to_datetime(df['uploadDate'])

# 카테고리 선택
category_list = sorted(df['category'].dropna().unique())
selected_category = st.selectbox("카테고리를 선택하세요", category_list)

# 날짜 범위 슬라이더
min_date = df['uploadDate'].min().date()
max_date = df['uploadDate'].max().date()
selected_date_range = st.date_input(
    "분석할 날짜 범위를 선택하세요",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
keyword_filter = st.text_input("🔍 특정 키워드를 필터링하려면 입력하세요", "")

top_n = st.selectbox("🔢 몇 개의 상위 키워드를 볼까요?", [10, 20, 30, 50], index=1)

start_date, end_date = selected_date_range
filtered_df = df[
    (df['category'] == selected_category) &
    (df['uploadDate'].dt.date >= start_date) &
    (df['uploadDate'].dt.date <= end_date)
]

if keyword_filter.strip():
    filtered_df = filtered_df[filtered_df['keyword'].str.contains(keyword_filter.strip(), case=False)]

# 타입 탭: flash / steady
tabs = st.tabs(["⚡ 반짝 키워드", "📈 꾸준히 상승 키워드"])

with tabs[0]:
    st.subheader(f"⚡ {selected_category} - 반짝 키워드 (TOP {top_n})")

    flash_df = filtered_df[filtered_df['type'] == 'flash'].copy()
    flash_df['numeric_score'] = pd.to_numeric(flash_df['score'], errors='coerce')
    flash_df['is_new'] = flash_df['score'].astype(str).str.upper() == 'NEW'

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

with tabs[1]:
    st.subheader(f"📈 {selected_category} - 꾸준히 상승 키워드 (TOP {top_n})")

    steady_df = filtered_df[filtered_df['type'] == 'steady'].copy()
    steady_df['numeric_score'] = pd.to_numeric(steady_df['score'], errors='coerce')
    steady_df = steady_df.sort_values("numeric_score", ascending=False).head(top_n)

    if not steady_df.empty:
        chart = alt.Chart(steady_df).mark_bar(size=20).encode(
            x=alt.X("numeric_score:Q", title="기울기"),
            y=alt.Y("keyword:N", sort='-x', title="키워드"),
            tooltip=["keyword", "score"]
        ).properties(height=500)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("선택한 기간 및 조건에 해당하는 꾸준 키워드가 없습니다.")
