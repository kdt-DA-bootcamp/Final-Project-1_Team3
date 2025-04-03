import requests
import os
import pandas as pd
from dotenv import load_dotenv
import signal
import sys

# API 키 로드
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
        elif response.status_code in [403, 400]:
            print(f"[!] 키 {key_index+1} 한도 초과. 다음 키로 전환.")
            key_index = (key_index + 1) % len(api_keys)
        else:
            print(f"[!] 기타 오류: {response.text}")
            break
    print("[X] 모든 API 키가 소진되었거나 요청 실패.")
    return None

# 중간 저장 파일 이름
filename = "video_categories.csv"

# 기존 데이터 불러오기 또는 빈 데이터프레임 초기화
if os.path.exists(filename):
    df_category = pd.read_csv(filename)
    collected_ids = set(df_category["videoID"].dropna().unique())
    print(f"기존 데이터 로드 완료: {len(collected_ids)}개 영상")
else:
    df_category = pd.DataFrame(columns=["videoID", "categoryID"])
    collected_ids = set()
    print("새로운 데이터 수집 시작")

def save_data():
    """중간 저장"""
    global df_category  # 전역 변수로 선언
    df_category.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"데이터 저장 완료: {filename}")

# 안전 종료 핸들러
def signal_handler(sig, frame):
    print("중단 신호 감지: 데이터 저장 중...")
    save_data()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_video_categories(video_ids):
    global df_category  # 전역 변수 사용 선언
    url = "https://www.googleapis.com/youtube/v3/videos"
    video_categories = []

    print(f"총 {len(video_ids)}개의 영상에 대해 카테고리 ID 수집 시작")

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        # 수집하지 않은 videoID만 필터링
        batch = [vid for vid in batch if vid not in collected_ids]
        if not batch:
            print(f"이미 수집된 ID - {i+1}번째 배치 건너뜀")
            continue

        print(f"{i+1}번째 배치 처리 중... (총 {len(batch)}개)")

        params = {
            "part": "snippet",
            "id": ",".join(batch),
        }
        response = get_valid_response(url, params)
        if not response:
            print(f"요청 실패 - {batch}")
            continue
        
        data = response.json()
        if not data.get("items"):
            print(f"⚠️ 응답에 영상 데이터 없음 - {batch}")
            continue
        
        print(f"응답 성공 - 수신 영상 수: {len(data.get('items', []))}")

        # 누락된 ID 확인
        returned_ids = {item["id"] for item in data.get("items", [])}
        missing_ids = set(batch) - returned_ids
        if missing_ids:
            print(f"응답에 누락된 videoID: {missing_ids}")

        for item in data.get("items", []):
            video_id = item.get("id")
            snippet = item.get("snippet", {})
            category_id = snippet.get("categoryId", "N/A")

            # 빈 category_id 방지
            if not category_id:
                print(f"카테고리 ID 없음: {video_id}")
                category_id = "N/A"
                
            video_categories.append([video_id, category_id])
            collected_ids.add(video_id)

        # 중간 저장
        if video_categories:
            new_df = pd.DataFrame(video_categories, columns=["videoID", "categoryID"])
            if df_category is not None and not df_category.empty:
                df_category = pd.concat([df_category, new_df]).drop_duplicates(subset=["videoID"], keep="last")
            else:
                df_category = new_df
            save_data()
            video_categories = []  # 중복 저장 방지
            print(f"중간 저장 완료: {filename}")
        else:
            print("수집된 데이터 없음 - 중간 저장 생략")

    print(f"카테고리 ID 수집 완료: 총 {len(df_category)}개")
    return video_categories

# 기존 CSV 파일에서 videoID 불러오기
video_df = pd.read_csv("videos_by_keywords_최종.csv")
video_ids = video_df["videoID"].dropna().unique().tolist()

# 카테고리 ID 가져오기
category_data = get_video_categories(video_ids)

# 최종 저장
save_data()
print("모든 데이터 수집 및 저장 완료")