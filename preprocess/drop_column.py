import pandas as pd

# 파일 경로에 맞게 CSV 파일 읽기
input_file = 'C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/new_category_data_no_pos.csv'
df = pd.read_csv(input_file)

# 원하는 열 삭제
df = df.drop(columns=['keyword', 'segment'])

# 결과 CSV 파일로 저장
output_file = 'C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/new_category_data_no_pos.csv'
df.to_csv(output_file, index=False)
print(f"열이 삭제된 파일이 '{output_file}'에 저장되었습니다.")
