import sys
import os
# 가장 먼저 상위 디렉토리를 모듈 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
from db.db_handler import process_and_upsert_channels, process_and_upsert_videos

def run_video_crawler():
    """video_crawler.py 실행 후, data 폴더에 CSV 파일이 생성됨"""
    try:
        print(">> video_crawler.py 실행 시작...")
        script_path = os.path.join(os.path.dirname(__file__), "video_crawler.py")
        subprocess.run(["python", script_path], check=True)
        print(">> video_crawler.py 실행 완료. CSV 파일이 data 폴더에 저장되었습니다.")
    except subprocess.CalledProcessError as e:
        print("!! video_crawler 실행 중 오류 발생:", e)
        sys.exit(1)

def run_db_handler():
    """db_handler.py의 함수들을 호출하여 CSV 파일로부터 DB에 데이터 적재"""
    try:
        print(">> 채널 데이터 DB 적재 시작...")
        process_and_upsert_channels("data/channels_by_keyword.csv")

        print(">> 영상 데이터 DB 적재 시작...")
        process_and_upsert_videos("data/videos_by_keyword.csv")

        print(">> DB 적재 작업 완료.")
    except Exception as e:
        print("!! DB 적재 작업 중 오류 발생:", e)
        sys.exit(1)

def main():
    # run_video_crawler()
    run_db_handler()

if __name__ == '__main__':
    main()