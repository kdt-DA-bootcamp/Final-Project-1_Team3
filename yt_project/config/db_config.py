import os
import pymysql
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# DB 연결 함수
def get_db_connection():
    """DB 연결 함수"""
    try:
        connection = pymysql.connect(
            host=os.getenv("MARIADB_HOST"),
            port=int(os.getenv("MARIADB_PORT")),
            user=os.getenv("MARIADB_USER"),
            password=os.getenv("MARIADB_PASSWORD"),
            database=os.getenv("MARIADB_DATABASE"),
            charset="utf8mb4"
        )
        print("DB 연결됨")
        return connection
    except pymysql.MySQLError as e:
        print(f"DB 연결불가: {e}")
        return None

# SQLAlchemy 엔진 생성
def get_db_engine():
    """SQLAlchemy 엔진 생성"""
    try:
        # 환경 변수 불러오기
        db_user = os.getenv("MARIADB_USER")
        db_password = os.getenv("MARIADB_PASSWORD")
        db_host = os.getenv("MARIADB_HOST")
        db_port = os.getenv("MARIADB_PORT")
        db_name = os.getenv("MARIADB_DATABASE")
        engine = create_engine(
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
        )
        print("엔진 생성 성공")
        return engine
    except Exception as e:
        print(f"엔진 생성 실패: {e}")
        return None
