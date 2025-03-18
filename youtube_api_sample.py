# 초기 설정: 라이브러리 설치
# pip install requests google-auth google-auth-oauthlib google-auth-httplib2
# pip install isodate

# 유튜브 API 설정(보안주의)
import requests
import isodate
import csv
import os
import re
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# API 키 가져오기
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3/"


# 해시태그 추출
def extract_hashtags(description):
    return list(set(re.findall(r"#\S+", description)))

# ISO 8601 duration 변환 (3분 이하 영상 제외)
def parse_duration(iso_duration):
    if not iso_duration:
        return "N/A"
    try:
        duration = isodate.parse_duration(iso_duration)
        duration_seconds = int(duration.total_seconds())

        if duration_seconds <= 180:
            return "SHORTS"
        return duration_seconds  # 정상적인 영상 길이 반환

    except Exception:
        return "N/A"

# 채널 정보
def get_channels_info(channel_ids):
    if not channel_ids:
        return {}

    url = f"{BASE_URL}channels?part=snippet,statistics&id={','.join(channel_ids)}&key={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        return {}

    data = response.json()
    channels_info = {}

    for item in data.get("items", []):
        channel_id = item.get("id", "Unknown")
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})

        channels_info[channel_id] = {
            "channelTitle": snippet.get("title", "Unknown"),
            "subscriberCount": int(statistics.get("subscriberCount", 0)),
            "viewCount": int(statistics.get("viewCount", 0)),
            "videoCount": int(statistics.get("videoCount", 0)),
        }

    return channels_info


# 한국 동영상 카테고리
def get_korea_video_categories():
    url = f"{BASE_URL}videoCategories?part=snippet&regionCode=KR&key={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    
    data = response.json()
    return [{"id": item.get("id", "Unknown"), "title": item.get("snippet", {}).get("title", "Unknown")} for item in data.get("items", [])]


# 특정 카테고리에서 동영상 리소스 가져오기 (쇼츠 제외 후 50개 유지)
def get_korea_videos_by_category(category_id, category_name, max_results=50, start_date=None, days=7):

    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_date_obj = datetime.now(timezone.utc) - timedelta(days=days)

    end_date_obj = start_date_obj + timedelta(days=days)
    
    published_after = start_date_obj.isoformat("T") + "Z"
    published_before = end_date_obj.isoformat("T") + "Z"

    videos = []
    channel_ids = set()
    next_page_token = None
    max_requests = 5  # 최대 5회 반복 요청 (API 과부하 방지)

    while len(videos) < max_results * 2 and max_requests > 0:  # 50개 이상 가져와서 필터링
        url = (f"{BASE_URL}videos?part=snippet,statistics,contentDetails&chart=mostPopular&regionCode=KR"
               f"&videoCategoryId={category_id}&maxResults=50&publishedAfter={published_after}&publishedBefore={published_before}&key={API_KEY}")

        if next_page_token:
            url += f"&pageToken={next_page_token}"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"동영상 데이터 요청 실패: {response.status_code}")
            return [], []

        data = response.json()
        if "items" not in data:
            break

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            content_details = item.get("contentDetails", {})

            duration_iso = content_details.get("duration", "PT0S")
            duration_seconds = parse_duration(duration_iso)

            # 쇼츠(3분 이하) 제외 (필터링 강화)
            if duration_seconds == "SHORTS" or duration_seconds == "N/A":
                continue  # 3분 이하 영상 또는 오류 발생 시 제외

            channel_id = snippet.get("channelId", "Unknown")
            channel_ids.add(channel_id)


            description = snippet.get("description", "")
            hashtags = extract_hashtags(description)
            video_tags = snippet.get("tags", [])
            merged_tags = list(set(hashtags + video_tags))

            video_info = {
                "id": item.get("id", "Unknown"),
                "categoryId": category_id,
                "categoryTitle": category_name,
                "channelId": channel_id,
                "channelTitle": snippet.get("channelTitle", "Unknown"),
                "title": snippet.get("title", "Unknown"),
                "viewCount": int(statistics.get("viewCount", 0)),  # 조회수 기준 정렬
                "likeCount": int(statistics.get("likeCount", 0)),
                "commentCount": int(statistics.get("commentCount", 0)),
                "publishedAt": snippet.get("publishedAt", "Unknown"),
                "duration": duration_seconds,  # 필터링 후 3분 이상 데이터만 저장
                "tags": merged_tags,
                "thumbnailUrl": snippet.get("thumbnails", {}).get("high", {}).get("url", "")
            }
            videos.append(video_info)

        next_page_token = data.get("nextPageToken", None)
        if not next_page_token:
            break  # 다음 페이지가 없으면 중단

        max_requests -= 1  # 요청 횟수 감소
    
    # 조회수 기준 정렬 후 상위 50개 선택
    sorted_videos = sorted(videos, key=lambda x: x["viewCount"], reverse=True)[:max_results]

    # 채널 정보 가져오기
    channels_info = get_channels_info(list(channel_ids))

    # 각 동영상에 채널 정보 추가
    for video in sorted_videos:
        channel_id = video["channelId"]
        channel_info = channels_info.get(channel_id, {
            "channelTitle": "Unknown",
            "subscriberCount": 0,
            "viewCount": 0,
            "videoCount": 0
        })
        video.update(channel_info)  # 채널 정보 병합

    return sorted_videos, list(channels_info.values())

# csv 파일 저장
def save_to_csv(directory, filename, data, headers):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    print(f'파일 저장 완료: {filepath}')


# 실행 함수
def get_korea_youtube_data(output_dir=".", start_date=None, days=7):
    categories = get_korea_video_categories()
    all_videos = []
    all_channels = []

    for category in categories:
        print(f"\n카테고리: {category['title']} (ID: {category['id']})")
        videos, channels = get_korea_videos_by_category(category["id"], category["title"], max_results=50, start_date=start_date, days=7)
        if videos:
            all_videos.extend(videos)
        if channels:
            all_channels.extend(channels)

    if all_videos:
        save_to_csv(output_dir, "korea_videos.csv", all_videos, 
        ["id", "categoryId", "categoryTitle", "channelId", "channelTitle", "title", 
         "viewCount", "likeCount", "commentCount", "publishedAt", "duration", "tags", "thumbnailUrl"])
        print("카테고리별 인기 동영상 정보 저장 완료: korea_videos.csv")
    
    if all_channels:
        save_to_csv(output_dir, "korea_channels.csv", all_channels, 
        ["channelId", "channelTitle", "subscriberCount", "viewCount", "videoCount"])
        print("채널 정보 저장 완료: korea_channels.csv")

# 실행 예시:  2025년 2월 4일부터 7일간의 데이터 가져오기
get_korea_youtube_data("./output", start_date="2025-02-04", days=7)