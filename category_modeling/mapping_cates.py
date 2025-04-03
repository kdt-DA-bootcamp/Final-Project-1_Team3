import pandas as pd

# 1. CSV 파일 불러오기
df_pre = pd.read_csv("preprocessed_data_comp.csv")
df_cat = pd.read_csv("video_categories.csv")

# 2. videoID 기준으로 병합 (왼쪽 조인: preprocessed_data_comp.csv의 모든 행 유지)
merged_df = pd.merge(df_pre, df_cat[['videoID', 'categoryID']], on='videoID', how='left')

# 병합 결과를 파일로 저장 (일치하는 videoID에 categoryID가 추가됨)
merged_output = "preprocessed_data_with_category.csv"
merged_df.to_csv(merged_output, index=False)
print(f"Merged file saved as {merged_output}")

# 3. preprocessed_data_comp.csv에는 있으나 video_categories.csv에는 없는 videoID 추출
#    병합 결과에서 categoryID가 NaN인 행이 해당됨
missing_df = merged_df[merged_df['categoryID'].isna()]
missing_output = "preprocessed_data_missing_video_categories.csv"
missing_df.to_csv(missing_output, index=False)
print(f"Missing videoIDs file saved as {missing_output}")
