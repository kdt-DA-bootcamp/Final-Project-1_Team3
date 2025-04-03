import pandas as pd
import ast
from nltk.tokenize import word_tokenize
import nltk

# NLTK 토큰화를 위한 필수 리소스 다운로드
nltk.download('punkt')

# CSV 파일 로드
file_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/filtered_video_data.csv"  # CSV 파일 경로
df = pd.read_csv(file_path)

# title 컬럼 토큰화
df["title_tokens"] = df["title"].apply(lambda x: word_tokenize(str(x)))

# tags 컬럼을 리스트로 변환 후 토큰화
def parse_tags(tag_string):
    try:
        tags_list = ast.literal_eval(tag_string)  # 문자열을 리스트로 변환
        return [tag for tag in tags_list] if isinstance(tags_list, list) else []
    except (ValueError, SyntaxError):
        return []

df["tags_tokens"] = df["tags"].apply(parse_tags)

# img_text 컬럼 토큰화
df['img_text_tokens'] = df["img_text"].apply(lambda x: word_tokenize(str(x)))

# 결과 확인
print(df[["title", "title_tokens", "tags", "tags_tokens", "img_text", "img_text_tokens"]].head())

# 결과를 새로운 CSV 파일로 저장
output_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/tokenized_filtered_data.csv"
df.to_csv(output_path, index=False)