import os
import pandas as pd
from google.cloud import vision
import pymysql
from config.db_config import get_db_connection



def update_thumbnail_texts(batch_size=16,
                           confidence_threshold=0.5, generic_stopwords=None):
    """
    DB의 video 테이블과 thumbnail 테이블을 join하여,
    thumbnail 테이블의 imgKw와 imgText가 비어있는 영상만 대상으로
    Google Cloud Vision API를 사용해 이미지 분석 결과를 업데이트합니다.
    
    만약 Vision API 응답에 에러가 있거나, 유의미한 키워드/텍스트가 추출되지 않을 경우
    기본 문자열 (예: "이미지 분석 실패")을 업데이트하여 이후 동일 URL을 다시 크롤링하지 않도록 합니다.
    
    Parameters:
        batch_size (int): 한 번에 처리할 이미지 개수 (기본 16)
        confidence_threshold (float): Vision API 결과를 사용할 최소 신뢰도 (기본 0.5)
        generic_stopwords (set): 필터링할 불용어 집합 (기본 {"Internet", "Image", "Photo caption"})
    """
    if generic_stopwords is None:
        generic_stopwords = {"Internet", "Image", "Photo caption"}

    # DB 연결
    conn = get_db_connection()
    if conn is None:
        print("DB 연결 실패")
        return
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # ───────────────────────────────────────────────────────────
    # 2) INSERT missing thumbnailURL rows
    insert_sql = """
    INSERT INTO thumbnail (thumbnailURL)
    SELECT DISTINCT v.thumbnailURL
    FROM video v
    LEFT JOIN thumbnail t
      ON v.thumbnailURL = t.thumbnailURL
    WHERE t.thumbnailURL IS NULL
      AND v.thumbnailURL IS NOT NULL
    """
    cursor.execute(insert_sql)
    conn.commit()
    print(">>> thumbnail 테이블에 없는 URL들 삽입 완료")
    # ───────────────────────────────────────────────────────────
    # 3) FK 제약 추가 (한 번만 실행하면 됩니다)
    try:
        fk_sql = """
        ALTER TABLE video
        ADD CONSTRAINT fk_video_thumbnail
          FOREIGN KEY (thumbnailURL)
          REFERENCES thumbnail(thumbnailURL)
          ON UPDATE CASCADE
          ON DELETE CASCADE
        """
        cursor.execute(fk_sql)
        conn.commit()
        print(">>> video.thumbnailURL 에 FK 제약 추가 완료")
    except Exception as e:
        # 이미 제약이 걸려 있으면 Duplicate 에러가 날 수 있으니 무시
        if 'Duplicate' in str(e) or 'exists' in str(e):
            print(">>> FK 제약이 이미 존재합니다. 건너뜁니다.")
        else:
            raise
    # ───────────────────────────────────────────────────────────
    # Google Cloud Vision client 생성 (서비스 계정 키는 환경변수로 설정됨)
    client = vision.ImageAnnotatorClient()

    # video와 thumbnail 테이블을 join하여 imgKw, imgText가 비어있는 데이터만 선택 (10건만)
    query = """
        SELECT v.thumbnailURL, v.uploadDate
        FROM video v
        JOIN thumbnail t ON v.thumbnailURL = t.thumbnailURL
        WHERE (t.imgKw IS NULL OR t.imgText IS NULL)
    """
    cursor.execute(query)
    data_from_db = cursor.fetchall()
    df = pd.DataFrame(data_from_db)

    if df.empty:
        print("업데이트할 대상이 없습니다.")
        cursor.close()
        conn.close()
        return

    # uploadDate를 datetime으로 변환 (날짜 조건이 없어도 저장된 날짜 포맷 맞춤)
    df['uploadDate'] = pd.to_datetime(df['uploadDate']).dt.tz_localize(None)

    print(f"총 {len(df)}개의 영상이 업데이트가 필요한 상태입니다.")

    # Vision API 요청을 위한 요청 리스트 구성
    image_requests = []
    indices = []
    for idx, row in df.iterrows():
        url = row['thumbnailURL']
        image = vision.Image(source=vision.ImageSource(image_uri=url))
        request = vision.AnnotateImageRequest(
            image=image,
            features=[
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
                vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
                vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
            ]
        )
        image_requests.append(request)
        indices.append(idx)

    # 기본 값 설정 (Vision API 결과가 없을 때 업데이트할 공백문자)
    default_imgKw = ""
    default_imgText = ""

    # 배치별로 Vision API 요청 및 DB 업데이트 수행
    for start_idx in range(0, len(image_requests), batch_size):
        batch_requests = image_requests[start_idx:start_idx+batch_size]
        batch_indices = indices[start_idx:start_idx+batch_size]

        response = client.batch_annotate_images(requests=batch_requests)
        responses = response.responses

        for idx, res in zip(batch_indices, responses):
            thumbnail_url = df.at[idx, 'thumbnailURL']

            # 응답에 error 메시지가 있으면 기본 값 사용
            if hasattr(res, 'error') and res.error.message:
                print(f"{thumbnail_url} 처리 중 에러 발생: {res.error.message}")
                keywords_str = default_imgKw
                detected_text = default_imgText
            else:
                # 라벨, 객체, 웹 엔터티 추출
                label_descriptions = [label.description for label in res.label_annotations if label.score >= confidence_threshold] if res.label_annotations else []
                object_names = [obj.name for obj in res.localized_object_annotations if obj.score >= confidence_threshold] if res.localized_object_annotations else []
                web_descriptions = [entity.description for entity in res.web_detection.web_entities if entity.description] if res.web_detection and res.web_detection.web_entities else []
                
                combined_keywords = []
                # 중복(대소문자 무시) 제거 및 불용어 필터링
                for keyword in label_descriptions + object_names + web_descriptions:
                    if keyword.lower() in {kw.lower() for kw in combined_keywords} or keyword in generic_stopwords:
                        continue
                    combined_keywords.append(keyword)
                keywords_str = ", ".join(combined_keywords)
                if not keywords_str:
                    keywords_str = default_imgKw

                # OCR 텍스트 추출
                detected_text = ""
                if res.text_annotations:
                    detected_text = ' '.join(res.text_annotations[0].description.split())
                if not detected_text:
                    detected_text = default_imgText

            # thumbnail 테이블 업데이트
            update_query = """
                UPDATE thumbnail
                SET imgKw = %s,
                    imgText = %s
                WHERE thumbnailURL = %s
            """
            cursor.execute(update_query, (keywords_str, detected_text, thumbnail_url))
        
        conn.commit()
        print(f"{min(start_idx+batch_size, len(image_requests))}/{len(image_requests)} 이미지 배치 처리 후 DB 업데이트 완료.")

    cursor.close()
    conn.close()
    print("모든 이미지 배치 처리가 완료되었습니다.")


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    # 서비스 계정 JSON 키 파일 경로 (환경변수에 설정)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\YH\Desktop\데이터분석가부트캠프\cloud_key\image-captioning-454201-197ede71ce67.json"
    # 날짜 범위 없이 전체 대상 중 조건에 맞는 10건만 처리
    update_thumbnail_texts()

# 터미널에서 다음의 코드를 타이핑 python -m crawler.thumbnail_crawler