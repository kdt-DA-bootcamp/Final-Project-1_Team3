import pandas as pd
import ast
from nltk.tokenize import word_tokenize
import nltk
from konlpy.tag import Okt

# NLTK 필수 리소스 다운로드
nltk.download('punkt')

# Okt 형태소 분석기 초기화
okt = Okt()

# CSV 파일 로드
file_path = "C:\\Users\\OWNER\\BootCamp\\project_Ytrend\\preprocessing\\videos_by_keywords_4.csv"  # CSV 파일 경로
df = pd.read_csv(file_path)

# title 컬럼 토큰화
df["title_tokens"] = df["title"].apply(lambda x: word_tokenize(str(x)))

# tags 컬럼을 리스트로 변환
def parse_tags(tag_string):
    try:
        tags_list = ast.literal_eval(tag_string)  # 문자열을 리스트로 변환
        return [tag for tag in tags_list] if isinstance(tags_list, list) else []
    except (ValueError, SyntaxError):
        return []

df["tags_tokens"] = df["tags"].apply(parse_tags)

# 품사 태깅 함수
def pos_tagging(tokens):
    return okt.pos(" ".join(tokens))  # 띄어쓰기 기반으로 하나의 문자열로 변환 후 태깅

# title_tokens에 대해 품사 태깅 수행
df["title_pos"] = df["title_tokens"].apply(pos_tagging)

# 결과 확인
print(df[["title", "title_tokens", "title_pos"]].head())