import pandas as pd

# 1. 파일 로드
tokenized_df = pd.read_csv("tokenized_filtered_data.csv")
pos_tagged_df = pd.read_csv("pos_tagged_data.csv")

# 2. pos_tagged_data에서 title을 키로, videoID를 값으로 하는 매핑 사전 생성
title_to_videoID = pos_tagged_df.set_index("title")["videoID"].to_dict()

# 3. tokenized_df에서 videoID가 "#NAME?"인 행을 찾아, 해당 title에 매핑된 videoID로 업데이트
mask = tokenized_df["videoID"] == "#NAME?"
tokenized_df.loc[mask, "videoID"] = tokenized_df.loc[mask, "title"].map(title_to_videoID)

# 4. 업데이트 결과 확인 (업데이트 후 videoID가 여전히 결측인 경우도 확인)
print("업데이트 후 결측 videoID:")
print(tokenized_df[tokenized_df["videoID"].isnull()])

# 5. 결과 저장
output_file = "tokenized_filtered_data_filled.csv"
tokenized_df.to_csv(output_file, index=False)
print(f"업데이트된 파일이 저장되었습니다: {output_file}")
