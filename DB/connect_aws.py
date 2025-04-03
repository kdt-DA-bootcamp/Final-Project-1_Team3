import os
from dotenv import load_dotenv
import pymysql

# .env 파일에 정의된 환경 변수 로드
load_dotenv(dotenv_path=".env.db")

# DB 연결 정보
# .env에서 환경 변수 가져오기
host = os.getenv('HOST')
port = int(os.getenv('PORT'))
user = os.getenv('USER')
password = os.getenv('PASSWORD')
database = os.getenv('DATABASE')

try:
    # DB 연결
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4"
    )
    print("MariaDB 연결 성공")
except pymysql.MySQLError as e:
    print(f"MariaDB 연결 실패: {e}")
finally:
    # 연결 종료
    if 'connection' in locals() and connection:
        connection.close()
        print("연결 종료")