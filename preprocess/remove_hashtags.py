import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("filtered_video_data.csv")

# 문자열 컬럼에서 '#' 제거 (모든 문자열 컬럼에 적용)
for col in ["title", "tags", "img_text"]:
    if col in df.columns:
        df[col] = df[col].str.replace("#", "", regex=False)


# 결과 확인 (상위 5개 행 출력)
print(df.head())

# 변경된 데이터 CSV 파일로 저장
df.to_csv("no_hashtags_data.csv", index=False)
