import pymysql

# DB 연결 정보
host = "3.36.132.60"  # 서버 IP 주소
port = 3308           # team3db 포트 번호
user = "team3"        # DB 사용자 이름
password = "test1234" # DB 비밀번호
database = "testdb"   # DB 이름

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

    cursor = connection.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    print(f"SELECT 1 결과: {result[0]}")

    while True:
        query = input("SQL> ")
        if query.lower() == "exit":
            break
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            for row in results:
                print(row)
        except Exception as e:
            print(f"쿼리 오류: {e}")

except pymysql.MySQLError as e:
    print(f"MariaDB 연결 실패: {e}")

finally:
    # 프로그램 종료 시 연결 닫기
    if 'connection' in locals() and connection:
        connection.close()
        print("연결 종료")