import pandas as pd
from config.db_config import get_db_engine, get_db_connection

# SQLAlchemy 엔진 생성
engine = get_db_engine()

def load_csv(file_path, encoding='utf-8'):
    """CSV 파일을 읽어서 데이터프레임 반환"""
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"CSV 파일 로드 실패: {e}")
        return pd.DataFrame()

def clean_video_df(df):
    """영상 데이터 전처리"""
    # 중복 제거
    df = df.drop_duplicates(subset='videoID')
    # uploadDate 변환
    try:
        df['uploadDate'] = pd.to_datetime(df['uploadDate'], errors='coerce', utc=True)
        df['uploadDate'] = df['uploadDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"날짜 변환 오류: {e}")
    # 컬럼명 정제 (최신 ERD 반영)
    df.columns = ['videoID', 'channelID', 'title', 'viewCount', 'likeCount', 'commentCount',
                  'uploadDate', 'duration', 'tags', 'thumbnailURL', 'categoryID']
    return df

def insert_channel_data(df):
    """채널 데이터 삽입 또는 업데이트"""
    conn = get_db_connection()
    if not conn:
        print("DB 연결 실패로 작업 중단")
        return
    try:
        with conn.begin():
            for _, row in df.iterrows():
                query = """
                INSERT INTO Channel (channelID, channelTitle, subscriberCount, totalViews, videosCount) 
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    channelTitle=VALUES(channelTitle), 
                    subscriberCount=VALUES(subscriberCount), 
                    totalViews=VALUES(totalViews), 
                    videosCount=VALUES(videosCount);
                """
                conn.execute(query, tuple(row))
            print("채널 데이터 삽입 또는 업데이트 완료")
    except Exception as e:
        print(f"채널 데이터 삽입 또는 업데이트 실패: {e}")
    finally:
        conn.close()

def upsert_video_data(df):
    """영상 데이터 삽입 또는 업데이트"""
    conn = get_db_connection()
    if not conn:
        print("DB 연결 실패로 작업 중단")
        return
    try:
        with conn.begin():
            for _, row in df.iterrows():
                query = """
                INSERT INTO Video (videoID, channelID, title, viewCount, likeCount, commentCount, 
                                   uploadDate, duration, tags, thumbnailURL, categoryID)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    channelID=VALUES(channelID), 
                    title=VALUES(title), 
                    viewCount=VALUES(viewCount), 
                    likeCount=VALUES(likeCount), 
                    commentCount=VALUES(commentCount), 
                    uploadDate=VALUES(uploadDate), 
                    duration=VALUES(duration), 
                    tags=VALUES(tags), 
                    thumbnailURL=VALUES(thumbnailURL), 
                    categoryID=VALUES(categoryID);
                """
                conn.execute(query, tuple(row))
            print("영상 데이터 삽입 또는 업데이트 완료")
    except Exception as e:
        print(f"영상 데이터 삽입 또는 업데이트 실패: {e}")
    finally:
        conn.close()

def fetch_from_db(query):
    """DB에서 데이터를 조회"""
    try:
        conn = get_db_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"데이터 조회 실패: {e}")
        return pd.DataFrame()

def process_and_upsert_channels(file_path):
    """CSV 파일을 불러와 채널 데이터를 DB에 삽입 또는 업데이트"""
    channel_df = load_csv(file_path)
    if not channel_df.empty:
        insert_channel_data(channel_df)

def process_and_upsert_videos(file_path):
    """CSV 파일을 불러와 전처리 후 DB에 삽입 또는 업데이트"""
    video_df = load_csv(file_path)
    if not video_df.empty:
        video_df = clean_video_df(video_df)
        # 유효한 채널 ID 필터링
        valid_channels = pd.read_sql("SELECT channelID FROM Channel", engine)
        valid_channel_ids = set(valid_channels['channelID'])
        video_df = video_df[video_df['channelID'].isin(valid_channel_ids)]
        print(f"삽입 가능한 영상 수: {len(video_df)}")
        upsert_video_data(video_df)