import requests
from dotenv import load_dotenv
import os

# API 키 가져오기
load_dotenv()
API_KEY = os.getenv("API_KEY")

def get_youtube_categories():
    url = "https://www.googleapis.com/youtube/v3/videoCategories"
    params = {
        "part": "snippet",
        "regionCode": "KR",
        "key": API_KEY,
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    categories = {}
    for item in data.get("items", []):
        category_id = item["id"]
        category_name = item["snippet"]["title"]
        categories[category_id] = category_name
    
    return categories

# 카테고리 출력
categories = get_youtube_categories()
print("📌 한국에서 제공하는 YouTube 카테고리 목록:")
for category_id, category_name in categories.items():
    print(f"🆔 {category_id}: {category_name}")