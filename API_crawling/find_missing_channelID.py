import pandas as pd

# CSV 파일 경로
videos_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/API_crawling/output/videos_by_keywords_최종(컬럼 정리).csv"
channels_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/total_channel_info.csv"

# 1. CSV 파일 불러오기 (깨진 줄 무시)
videos_df = pd.read_csv(videos_path, on_bad_lines='skip')  # pandas 1.3 이상
channels_df = pd.read_csv(channels_path)

# 2. 각각의 channelID 추출
video_channel_ids = set(videos_df['channelID'].dropna().unique())
channel_info_ids = set(channels_df['channelID'].dropna().unique())

# 3. 누락된 channelID 찾기
missing_channel_ids = list(video_channel_ids - channel_info_ids)

# 4. 결과를 데이터프레임으로 저장
missing_df = pd.DataFrame(missing_channel_ids, columns=["missing_channelID"])

# 5. CSV로 저장
missing_df.to_csv("missing_channelIDs.csv", index=False)

print("✅ 누락된 채널 ID 목록 저장 완료: missing_channelIDs.csv")
