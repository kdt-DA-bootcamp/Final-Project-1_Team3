import requests
import isodate
import csv
import os
import re
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd

# API 키 가져오기
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 사용자 입력: 시작 날짜 설정 (예: "2025-02-02", "241005", "20241005")
user_input_date = "2025-02-02"

# 입력 날짜 형식을 자동 변환하는 함수
def parse_date(input_date):
    try:
        if "-" in input_date:  # YYYY-MM-DD 형식
            return datetime.strptime(input_date, "%Y-%m-%d")
        elif len(input_date) == 6:  # YYMMDD 형식
            return datetime.strptime(input_date, "%y%m%d")
        elif len(input_date) == 8:  # YYYYMMDD 형식
            return datetime.strptime(input_date, "%Y%m%d")
        else:
            raise ValueError("올바른 날짜 형식이 아닙니다.")
    except ValueError as e:
        print(f"❌ 날짜 변환 오류: {e}")
        return None

# 시작 날짜 변환
start_date_obj = parse_date(user_input_date)
if start_date_obj:
    start_date = start_date_obj.replace(tzinfo=timezone.utc).isoformat()
    end_date = (start_date_obj + timedelta(days=7)).replace(tzinfo=timezone.utc).isoformat()
else:
    raise ValueError("날짜 형식이 잘못되었습니다. 올바른 형식(YYYY-MM-DD, YYMMDD, YYYYMMDD)으로 입력하세요.")

# 해시태그 추출하기
def extract_hashtags(description):
    hashtags = re.findall(r"#\w+", description)  # "#문자열" 패턴 찾기
    return [tag.strip("#") for tag in hashtags]  # "#" 제거 후 리스트로 반환

# ISO 8601 형식의 duration을 초 단위로 변환
def convert_duration_to_seconds(duration):
    try:
        parsed_duration = isodate.parse_duration(duration)
        return int(parsed_duration.total_seconds()) if parsed_duration else 0
    except:
        return 0

# 한국에서 제공하는 카테고리 ID 리스트
categories = {1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
         17: "Sports", 18: "Short Movies", 19: "Travel & Events", 20: "Gaming", 21: "Videoblogging",
         22: "People & Blogs", 23: "Comedy", 24: "Entertainment", 25: "News & Politics",
         26: "Howto & Style", 27: "Education", 28: "Science & Technology", 30: "Movies",
         31: "Anime/Animation", 32: "Action/Adventure", 33: "Classics", 34: "Comedy", 35: "Documentary",
         36: "Drama", 37: "Family", 38: "Foreign", 39: "Horror", 40: "Sci-Fi/Fantasy", 41: "Thriller",
         42: "Shorts", 43: "Shows", 44: "Trailers"}


# 카테고리별 인기 영상 가져오기
def get_popular_videos(category_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics,contentDetails",
        "chart": "mostPopular",
        "regionCode": "KR",
        "videoCategoryId": category_id,
        "maxResults": 50,
        "key": API_KEY,
    }
    
    response = requests.get(url, params=params)
    return response.json()


# 채널 정보 가져오기
def get_channel_info(channel_id):
    url = "https://www.googleapis.com/youtube/v3/channels"
    params = {
        "part": "snippet,statistics",
        "id": channel_id,
        "key": API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json()

video_data = []
channel_data = {}

# 카테고리별 인기 영상 정보 수집
for category_id in categories.keys():
    print(f"📌 카테고리 ID {category_id} ({categories[category_id]})의 인기 영상 가져오는 중...")

    videos = get_popular_videos(category_id)
    
    for video in videos.get("items", []):
        video_id = video["id"]
        snippet = video["snippet"]
        statistics = video.get("statistics", {})
        content_details = video.get("contentDetails", {})
        
        title = snippet["title"]
        description = snippet.get("description", "")
        channel_id = snippet["channelId"]
        upload_date = snippet.get("publishedAt", "")
        thumbnail_url = snippet["thumbnails"]["high"]["url"]

        # 해시태그 추출 후 기존 태그 목록과 합치기
        extracted_tags = extract_hashtags(description)
        tags = snippet.get("tags", [])
        tags = list(set(tags + extracted_tags))  # 중복 제거

        view_count = statistics.get("viewCount", "0")
        like_count = statistics.get("likeCount", "0")
        comment_count = statistics.get("commentCount", "0")
        duration = convert_duration_to_seconds(content_details.get("duration", "PT0S"))

        # 영상 정보 저장
        video_data.append([
            video_id, category_id, channel_id, title,
            view_count, like_count, comment_count, upload_date, duration, 
            tags, thumbnail_url
        ])

        # 채널 정보가 없으면 가져오기
        if channel_id not in channel_data:
            channel_info = get_channel_info(channel_id)
            if "items" in channel_info and len(channel_info["items"]) > 0:
                channel_item = channel_info["items"][0]
                channel_snippet = channel_item["snippet"]
                channel_statistics = channel_item["statistics"]

                channel_name = channel_snippet["title"]
                subscriber_count = channel_statistics.get("subscriberCount", "0")
                total_views = channel_statistics.get("viewCount", "0")
                video_count = channel_statistics.get("videoCount", "0")

                # 채널 정보 저장
                channel_data[channel_id] = [channel_id, channel_name, subscriber_count, total_views, video_count]


# 데이터프레임 생성 및 CSV 저장
video_df = pd.DataFrame(video_data, columns=[
    "videoID", "categoryID", "channelID", "title",
    "viewCount", "likeCount", "commentCount", "updateDate", "duration",
    "tags", "thumbnailURL"
])
channel_df = pd.DataFrame(list(channel_data.values()), columns=[
    "channelID", "channelTitle", "subscriberCount", "totalViews", "videosCount"
])

# CSV 파일로 저장
video_df.to_csv("youtube_videos.csv", index=False, encoding="utf-8-sig")
channel_df.to_csv("youtube_channels.csv", index=False, encoding="utf-8-sig")

print("✅ 유튜브 인기 영상 및 채널 정보 저장 완료!")