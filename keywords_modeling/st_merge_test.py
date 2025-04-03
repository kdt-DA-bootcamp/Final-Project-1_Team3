import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# ------------------- 피클 파일 로드 -------------------
@st.cache_data
def load_results(filename="keyword_trends.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"파일을 불러오는 중 오류 발생: {e}")
        return {}

results = load_results()

# ------------------- 스트림릿 UI -------------------
st.title("꾸준히 상승 중인 키워드 분석")
st.markdown("""
최근 데이터 분석을 통해 **꾸준히 상승하는 키워드**를 추출했습니다.  
카테고리를 선택하여 주요 키워드와 그 추세를 확인해보세요.
""")

# ------------------- 카테고리 선택 -------------------
category_map = {
    '1': '엔터테인먼트', '2': '차량', '3': '여행', '4': '게임', '5': '스포츠',
    '6': '라이프', '7': '정치', '8': '반려동물', '9': '교육/Howto', '10': '과학/기술'
}

categories = list(results.keys())
selected_category = st.selectbox("카테고리 선택", categories, format_func=lambda x: category_map.get(x, "알 수 없음"))

# ------------------- 데이터 확인 -------------------
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

        # ------------------- 상승 추세 시각화 -------------------
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