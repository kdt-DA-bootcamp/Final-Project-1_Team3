import streamlit as st

# 이미지와 videoID 매핑
video_data = [
    {
        "image_url": "https://i.ytimg.com/vi/c3VJbVlItOs/hqdefault.jpg",
        "video_id": "c3VJbVlItOs"
    },
    {
        "image_url": "https://i.ytimg.com/vi/Gq6HLyy1k6g/hqdefault.jpg",
        "video_id": "Gq6HLyy1k6g"
    },
    {
        "image_url": "https://i.ytimg.com/vi/bnl5QrzQ3F0/hqdefault.jpg",
        "video_id": "bnl5QrzQ3F0"
    }
]

st.title("썸네일 이미지")

# 이미지 클릭 시 유튜브 비디오 페이지로 이동하는 링크 생성
for item in video_data:
    image_url = item["image_url"]
    video_id = item["video_id"]
    
    # videoID를 이용해 원본 비디오 URL 생성 (유튜브 기준)
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # HTML 코드 생성
    image_html = f'<a href="{video_url}" target="_blank"><img src="{image_url}" width="300"></a>'
    
    # Streamlit에서 HTML 렌더링
    st.markdown(image_html, unsafe_allow_html=True)