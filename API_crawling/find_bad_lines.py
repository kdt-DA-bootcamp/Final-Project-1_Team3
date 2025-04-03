import pandas as pd

file_path = "videos_by_keywords_최종.csv"

# 파일 전체를 문자열로 읽기 (에러 무시)
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 기준이 되는 컬럼 개수 (보통 헤더 줄)
expected_columns = len(lines[0].strip().split(','))

# 깨진 줄 찾기
broken_lines = []
for idx, line in enumerate(lines):
    column_count = len(line.strip().split(','))
    if column_count != expected_columns:
        broken_lines.append((idx + 1, column_count, line.strip()))

# 결과 보기
for line_num, col_count, content in broken_lines[:10]:  # 처음 10개만 출력
    print(f"❌ Line {line_num}: 열 {col_count}개 → {content[:100]}...")

print(f"\n총 깨진 줄 수: {len(broken_lines)}")