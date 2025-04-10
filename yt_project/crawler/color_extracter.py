import os
import requests
from io import BytesIO
from PIL import Image
import extcolors
import pymysql
import json
from dotenv import load_dotenv
from config.db_config import get_db_connection  # DB 접속 함수를 가져옵니다.

# .env 파일의 환경 변수 로드 (필요한 경우)
load_dotenv()

def extract_colors_from_url(url, tolerance=12, limit=5):
    """
    주어진 URL의 이미지를 다운로드하여, extcolors 라이브러리를 사용해 색상 정보를 추출합니다.
    반환 값은 각 색상의 (RGB, 픽셀 수, 퍼센트) 튜플 리스트입니다.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # 오류 발생 시 예외 처리
        img = Image.open(BytesIO(response.content))
        
        # 이미지에서 색상 추출: colors는 [((R, G, B), 픽셀수), ...]
        colors, pixel_count = extcolors.extract_from_image(img, tolerance=tolerance, limit=limit)
        total = sum(count for (rgb, count) in colors)
        colors_with_percentage = [
            (rgb, count, round((count / total * 100), 2)) for (rgb, count) in colors
        ]
        return colors_with_percentage
    except Exception as e:
        print(f"URL 처리 중 오류 발생: {url}\n오류 메시지: {e}")
        return None

def update_colorpalette(tolerance=12, limit=5, batch_size=10):
    """
    DB의 thumbnail 테이블에서 thumbnailURL을 읽어와 각 이미지의 색상 정보를 추출하고,
    추출한 결과를 JSON 문자열로 변환하여 DB의 colorpalette 컬럼을 업데이트합니다.
    
    대상 레코드는 colorpalette가 NULL 또는 빈 문자열인 경우이며,
    추출 실패 시 기본 문자열("색상 추출 실패")을 저장하여 오류 반복 크롤링을 방지합니다.
    
    배치 처리로 일정 건수마다 커밋합니다.
    """
    # DB 연결 생성
    conn = get_db_connection()
    if conn is None:
        print("DB 연결 실패")
        return

    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # 업데이트할 대상 레코드 선별 (colorpalette 컬럼이 NULL 또는 빈 문자열인 경우)
    select_query = "SELECT thumbnailURL FROM thumbnail WHERE (colorpalette IS NULL)"
    cursor.execute(select_query)
    rows = cursor.fetchall()

    if not rows:
        print("업데이트할 대상이 없습니다.")
        cursor.close()
        conn.close()
        return

    print(f"총 {len(rows)}개의 이미지에 대해 색상 정보를 추출합니다.")

    default_color = ""  # 색상 추출 실패 시 사용할 공백문자
    count = 0

    for row in rows:
        url = row['thumbnailURL']
        colors = extract_colors_from_url(url, tolerance=tolerance, limit=limit)
        # 색상 정보 추출에 실패한 경우 default_color를 JSON 문자열로 저장
        colors_json = json.dumps(colors) if colors is not None else json.dumps(default_color)

        update_query = """
            UPDATE thumbnail
            SET colorpalette = %s
            WHERE thumbnailURL = %s
        """
        cursor.execute(update_query, (colors_json, url))
        count += 1

        # 배치 처리: 지정 건수마다 커밋
        if count % batch_size == 0:
            conn.commit()
            print(f"{count}개의 이미지 업데이트 완료.")

    conn.commit()
    print("모든 이미지의 색상 추출 및 DB 업데이트가 완료되었습니다.")
    cursor.close()
    conn.close()

if __name__ == '__main__':
    # tolerance와 limit 값을 필요에 따라 조정 가능
    update_colorpalette(tolerance=12, limit=5)