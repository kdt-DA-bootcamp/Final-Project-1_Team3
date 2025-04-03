import pandas as pd

# 1. 원본 CSV 파일 불러오기
df = pd.read_csv("merged_c_c_t_NN_T.csv")

# 2. 필요한 컬럼만 선택
columns_to_keep = ["videoID", "title", "tags", "img_text"]
filtered_df = df[columns_to_keep]

# 3. 새로운 CSV로 저장
filtered_df.to_csv("filtered_video_data.csv", index=False, encoding="utf-8-sig")

print("✅ 필요한 컬럼만 추출하여 filtered_video_data.csv로 저장 완료!")
