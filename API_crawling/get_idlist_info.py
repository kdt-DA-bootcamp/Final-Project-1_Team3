import os
import csv
import time
import requests
from dotenv import load_dotenv
from more_itertools import chunked  # pip install more-itertools
import pandas as pd

# 📌 1. .env에서 API 키 불러오기
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
        print(f"🔁 API 키 전환 → {api_index + 1}번째 키 사용")
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

# 📌 2. 수집할 채널 ID 리스트 정의
channel_ids = [
    "UCOWK2xUc1c6fdytaJnETnLg"
]

# 📌 3. 저장 경로 지정
output_path = "channel_info_result.csv"

# 📌 4. 결과 저장
with open(output_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["channelID", "channelTitle", "subscriberCount", "totalViews", "videosCount"])
    writer.writeheader()

    for i, batch in enumerate(chunked(channel_ids, 50)):
        print(f"🔍 Batch {i+1}: {len(batch)}개 채널 요청 중...")
        try:
            results = get_channel_infos_batch(batch)
            for info in results:
                writer.writerow(info)
            f.flush()
        except RuntimeError as e:
            print(str(e))
            break
        time.sleep(1.1)  # Rate limit 방지

print(f"\n✅ 완료! 결과 저장 위치: {output_path}")
