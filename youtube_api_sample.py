# 초기 설정: 라이브러리 설치
# pip install requests google-auth google-auth-oauthlib google-auth-httplib2

# 유튜브 API 설정(보안주의)
import requests
import json
import csv

API_KEY = "AIzaSyCVSFqv9eMFoQWrWEwOTt5RVS20iBCNMSM"

BASE_URL = "https://www.googleapis.com/youtube/v3/"

# 조회 조건 설정하기
MIN_SUBSCRIBERS = 100000
MIN_VIEWS = 1000000

# i18Regions에서 'KR'이 존재하는지 확인
def check_korea_region():
    url = f"{BASE_URL}i18nRegions?part=snippet&key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    for item in data.get("items", []):
        if item["snippet"]["gl"] == "KR":
            return True
    return False


# 채널 통계 정보 가져오기
def get_channel_statistics(channel_id):
    url = f"{BASE_URL}channels?part=statistics&id={channel_id}&key={API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    if not data.get("items"):
        return None
    
    stats = data["items"][0]["statistics"]
    return {
        "subscriberCount": int(stats.get("subscriberCount", 0)),
        "viewCount": int(stats.get("viewCount", 0))
    }


# 한국 채널 리소스 가져오기(구독자 수 필터 포함)
def get_korea_channels():
    url  = f"{BASE_URL}search?part=snippet&type=channel&regionCode=KR&maxResults=5&key={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"채널 데이터 요청 실패: {response.status_code}")
        return []
    
    data = response.json()
    channels = []

    for item in data.get("items", []):
        channel_id = item["id"]["channelId"]
        channel_data = get_channel_statistics(channel_id)

        if channel_data and channel_data["subscriberCount"] >= MIN_SUBSCRIBERS:
            channel_info = {
                "id": channel_id,
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                "subscribers": channel_data["subscriberCount"],
                "totalViews": channel_data["viewCount"]
            }
            channels.append(channel_info)
    
    return channels

# 동영상 리소스 가져오기
def get_korea_videos():
    url = f"{BASE_URL}videos?part=snippet,statistics&chart=mostPopular&regionCode=KR&maxResults=5&key={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"동영상 데이터 요청 실패: {response.status_code}")
        return []
    
    data = response.json()
    videos = []

    for item in data.get("items", []):
        view_count = int(item["statistics"].get("viewCount", 0))
        
        if view_count >= MIN_VIEWS:
            video_info = {
                "id": item["id"],
                "title": item["snippet"]["title"],
                "channelTitle": item["snippet"]["channelTitle"],
                "viewCount": view_count,
                "likeCount": int(item["statistics"].get("likeCount", 0)),
                "commentCount": int(item["statistics"].get("commentCount", 0)),
                "publishedAt": item["snippet"]["publishedAt"],
                "thumbnailUrl": item["snippet"]["thumbnails"]["high"]["url"]
            }
            videos.append(video_info)

    return videos


# # 한국에서 제공하는 동영상 모든 카테고리 가져오기
# def get_korea_video_categories():
#     url = f"{BASE_URL}videoCategories?part=snippet&regionCode=KR&key={API_KEY}"
#     response = requests.get(url)
#     data = response.json()

#     categories = []
#     for item in data.get("items", []):
#         category = {
#             "id": item["id"], 
#             "title": item["snippet"]["title"], 
#             "assignable": item["snippet"]["assignable"]
#         }
#         categories.append(category)

#     return categories

# 데이터 csv 파일 저장
def save_to_csv(filename, data, headers):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

# 실행 함수
def get_korea_youtube_data():
    if not check_korea_region():
        print("한국(KR) 지역이 지원되지 않습니다.")
        return

    print("\n한국 인기 채널 목록: ")
    korea_channels = get_korea_channels()
    for ch in korea_channels:
        print(ch)

    print("\n한국 인기 동영상 목록: ")
    korea_videos = get_korea_videos()
    for video in korea_videos:
        print(video)

    # print("\n한국 동영상 카테고리 목록: ")
    # korea_categories = get_korea_video_categories()
    # for category in korea_categories:
    #     print(category)

    # CSV 파일로 저장
    if korea_channels:
        save_to_csv("korea_channels.csv", korea_channels, 
                    ["id", "title", "description", "thumbnail", "subscribers", "totalViews"])
        print("📁 채널 정보 저장 완료: korea_channels.csv")

    if korea_videos:
        save_to_csv("korea_videos.csv", korea_videos, 
                    ["id", "title", "channelTitle", "viewCount", "likeCount", "commentCount", "publishedAt", "thumbnailUrl"])
        print("📁 동영상 정보 저장 완료: korea_videos.csv")

# 실행
get_korea_youtube_data()