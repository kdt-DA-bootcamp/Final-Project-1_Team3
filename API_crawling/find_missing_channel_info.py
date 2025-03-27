import os
import time
import csv
import requests
import pandas as pd
from dotenv import load_dotenv
from more_itertools import chunked  # pip install more-itertools

# 1. .env에서 API 키 불러오기
load_dotenv()
api_keys = [
    os.environ[key]
    for key in sorted(
        [k for k in os.environ if k.startswith("API_KEY_")],
        key=lambda x: int(x.split("_")[2]) if x.split("_")[2].isdigit() else 0
    )
]
api_index = 0

def get_api_key():
    global api_index
    if api_index >= len(api_keys):
        raise RuntimeError("❌ 모든 API 키의 할당량이 초과되었습니다.")
    return api_keys[api_index]

def switch_api_key():
    global api_index
    api_index += 1
    if api_index < len(api_keys):
        print(f"🔁 API 키 전환 → {api_index + 1}번째 키")
    else:
        raise RuntimeError("❌ 더 이상 사용할 수 있는 API 키가 없습니다.")

def get_channel_infos_batch(channel_ids):
    while True:
        api_key = get_api_key()
        ids = ",".join(channel_ids)
        url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&id={ids}&key={api_key}"
        response = requests.get(url)

        if response.status_code == 403:
            error_reason = response.json().get("error", {}).get("errors", [{}])[0].get("reason", "")
            if error_reason in ["quotaExceeded", "userRateLimitExceeded"]:
                print(f"⚠️ API 키 {api_index + 1} 할당량 초과")
                switch_api_key()
                continue
            else:
                print(f"❌ 접근 오류: {response.text}")
                return []
        elif response.status_code != 200:
            print(f"❌ 기타 오류: {response.status_code} - {response.text}")
            return []

        results = []
        for item in response.json().get("items", []):
            snippet = item['snippet']
            stats = item['statistics']
            results.append({
                "channelID": item['id'],
                "channelTitle": snippet.get('title', ''),
                "subscriberCount": stats.get('subscriberCount', '0'),
                "totalViews": stats.get('viewCount', '0'),
                "videosCount": stats.get('videoCount', '0')
            })
        return results

# 2. 누락된 ID만 필터링
input_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/missing_channelIDs_2.csv"
output_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/missing_channels2_info.csv"

# 전체 수집 대상
missing_df = pd.read_csv(input_path)
target_ids = set(missing_df['missing_channelID'].dropna().unique())

# 이미 수집된 것 불러오기
if os.path.exists(output_path):
    collected_df = pd.read_csv(output_path)
    collected_ids = set(collected_df['channelID'].dropna().unique())
else:
    collected_ids = set()
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["channelID", "channelTitle", "subscriberCount", "totalViews", "videosCount"])
        writer.writeheader()

# 남은 것만 처리
remaining_ids = list(target_ids - collected_ids)
print(f"🧩 수집되지 않은 채널 수: {len(remaining_ids)}")

# 3. 배치 단위로 API 호출
with open(output_path, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["channelID", "channelTitle", "subscriberCount", "totalViews", "videosCount"])
    for i, batch in enumerate(chunked(remaining_ids, 50)):
        print(f"🔍 Batch {i+1}/{len(remaining_ids)//50 + 1} → {len(batch)}개 채널 요청")
        try:
            results = get_channel_infos_batch(batch)
            for info in results:
                writer.writerow(info)
            f.flush()
        except RuntimeError as e:
            print(str(e))
            break
        time.sleep(1.1)  # 속도 제어
