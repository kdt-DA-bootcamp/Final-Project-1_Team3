import subprocess
import sys

def run_thumbnail_crawler():
    """thumbnail_crawler.py 실행 후, 결과 메시지 출력"""
    try:
        print(">> thumbnail_crawler.py 실행 시작...")
        subprocess.run(["python", "-m", "crawler.thumbnail_crawler"], check=True)
        print(">> thumbnail_crawler.py 실행 완료.")
    except subprocess.CalledProcessError as e:
        print("!! thumbnail_crawler 실행 중 오류 발생:", e)
        sys.exit(1)

def run_color_extractor():
    """color_extracter.py 실행 후, 결과 메시지 출력"""
    try:
        print(">> color_extracter.py 실행 시작...")
        subprocess.run(["python", "-m", "crawler.color_extracter"], check=True)
        print(">> color_extracter.py 실행 완료.")
    except subprocess.CalledProcessError as e:
        print("!! color_extracter 실행 중 오류 발생:", e)
        sys.exit(1)

def main():
    run_thumbnail_crawler()
    run_color_extractor()

if __name__ == '__main__':
    main()