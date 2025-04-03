import requests
import isodate
import os
import re
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# API 키 여러 개 로드
load_dotenv()

def load_api_keys():
    keys = []
    i = 1
    while True:
        key = os.getenv(f"API_KEY{i}")
        if key:
            keys.append(key)
            i += 1
        else:
            break
    return keys

api_keys = load_api_keys()
key_index = 0

def get_valid_response(url, params):
    global key_index
    max_attempts = len(api_keys)
    for _ in range(max_attempts):
        params["key"] = api_keys[key_index]
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response
        elif response.status_code in [403, 400]:  # 할당량 초과 등
            print(f"[!] 키 {key_index+1} 한도 초과. 다음 키로 전환.")
            key_index = (key_index + 1) % len(api_keys)
        else:
            print(f"[!] 기타 오류: {response.text}")
            break
    print("[X] 모든 API 키가 소진되었거나 요청 실패.")
    return None

# 기준 날짜 설정 (UTC 기준 현재시간)
current_date_obj = datetime.now(timezone.utc)
one_month_ago = current_date_obj - timedelta(days=30)
start_date = one_month_ago.isoformat()
end_date = current_date_obj.isoformat()

# 키워드 불러오기
keywords_df = pd.read_csv("카테고리별 키워드 리스트.csv")
keywords = keywords_df['keywords'].dropna().unique().tolist()

def convert_duration_to_seconds(duration):
    try:
        parsed_duration = isodate.parse_duration(duration)
        seconds = int(parsed_duration.total_seconds()) if parsed_duration else 0
        return seconds
    except Exception as e:
        print(f"Duration 변환 오류: {e}")
        return 0

def contains_korean(text):
    return bool(re.search(r"[가-힣]", text))

def search_video_ids_by_keyword(keyword):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_ids = set()

    period = timedelta(days=5)
    start_dt = datetime.now(timezone.utc) - timedelta(days=30)
    end_dt = datetime.now(timezone.utc)

    while start_dt < end_dt:
        part_start = start_dt.isoformat()
        part_end = (start_dt + period).isoformat()
        print(f"- {part_start[:10]} ~ {part_end[:10]} 검색 중...")

        params = {
            "part": "snippet",
            "type": "video",
            "regionCode": "KR",
            "order": "viewCount",
            "maxResults": 50,
            "publishedAfter": part_start,
            "publishedBefore": part_end,
            "q": keyword,
            "relevanceLanguage": "ko",
        }

        next_page_token = None
        while True:
            if next_page_token:
                params["pageToken"] = next_page_token
            response = get_valid_response(search_url, params)
            if not response:
                break
            data = response.json()
            for item in data.get("items", []):
                vid = item.get("id", {}).get("videoId")
                title = item.get("snippet", {}).get("title", "")
                if vid and contains_korean(title):
                    video_ids.add(vid)
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        start_dt += period

    return list(video_ids)

def get_videos_details(video_ids):
    url = "https://www.googleapis.com/youtube/v3/videos"
    all_videos = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(batch),
        }
        response = get_valid_response(url, params)
        if not response:
            continue
        data = response.json()
        all_videos.extend(data.get("items", []))
    return all_videos

def get_channels_info(channel_ids):
    url = "https://www.googleapis.com/youtube/v3/channels"
    all_channels = []
    channel_ids = list(channel_ids)
    missing_ids = []

    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch),
        }
        response = get_valid_response(url, params)
        if not response:
            print(f"채널 조회 실패: {batch}")
            missing_ids.extend(batch)
            continue
        data = response.json()
        all_channels.extend(data.get("items", []))

        # 누락된 채널 ID 확인 (API 응답에 없는 경우)
        returned_ids = {item["id"] for item in data.get("items", [])}
        batch_missing = set(batch) - returned_ids
        if batch_missing:
            print(f"[!] 응답에 누락된 채널 ID: {batch_missing}")
            missing_ids.extend(batch_missing)

    if missing_ids:
        print(f"[!] 총 누락된 채널 수: {len(missing_ids)} → 재조회 시도")
        for i in range(0, len(missing_ids), 50):
            retry_batch = missing_ids[i:i+50]
            params = {
                "part": "snippet,statistics",
                "id": ",".join(retry_batch),
            }
            response = get_valid_response(url, params)
            if not response:
                continue
            data = response.json()
            all_channels.extend(data.get("items", []))

    return all_channels

# 수집 시작
video_data = []
channel_ids = set()
seen_video_ids = set()

print(f"\n⚡ 키워드 '{keyword}' 수집 중...")
video_ids = search_video_ids_by_keyword(keyword)
print(f" - 검색된 영상 수: {len(video_ids)}")
if video_ids:
    videos_details = get_videos_details(video_ids)
    for video in videos_details:
        video_id = video.get("id")
        if video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)

        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})
        content_details = video.get("contentDetails", {})

        channel_id = snippet.get("channelId", "")
        title = snippet.get("title", "")
        view_count = int(statistics.get("viewCount", "0"))
        like_count = statistics.get("likeCount", "0")
        comment_count = statistics.get("commentCount", "0")
        upload_date = snippet.get("publishedAt", "")
        duration_sec = convert_duration_to_seconds(content_details.get("duration", "PT0S"))
        thumbnail_url = snippet.get("thumbnails", {}).get("high", {}).get("url", "")
        tags = snippet.get("tags", [])

        video_data.append([
            video_id, channel_id, title, view_count, like_count, comment_count,
            upload_date, duration_sec, tags, thumbnail_url, keyword
        ])
        channel_ids.add(channel_id)

channel_data = []
channels_details = get_channels_info(channel_ids)
seen_channel_ids = set()
for channel in channels_details:
    channel_id = channel.get("id")
    if channel_id in seen_channel_ids:
        continue
    seen_channel_ids.add(channel_id)

    snippet = channel.get("snippet", {})
    statistics = channel.get("statistics", {})
    channel_data.append([
        channel_id, snippet.get("title", ""), statistics.get("subscriberCount", "0"),
        statistics.get("viewCount", "0"), statistics.get("videoCount", "0")
    ])

# 저장 시 쉼표 이슈 방지 (모든 필드 큰따옴표 감싸기)
df_videos = pd.DataFrame(video_data, columns=[
    "videoID", "channelID", "title", "viewCount", "likeCount", "commentCount",
    "uploadDate", "duration", "tags", "thumbnailURL", "keyword"
])
df_videos.to_csv("videos_by_keyword.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
print("영상 데이터 저장 완료: videos_by_keyword.csv")

df_channels = pd.DataFrame(channel_data, columns=[
    "channelID", "channelTitle", "subscriberCount", "totalViews", "videosCount"
])
df_channels.to_csv("channels_by_keyword.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
print("채널 데이터 저장 완료: channels_by_keyword.csv")