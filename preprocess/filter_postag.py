import pandas as pd
import ast
from nltk.tokenize import word_tokenize
import nltk
from konlpy.tag import Okt

# # NLTK 필수 리소스 다운로드
# nltk.download('punkt')

# Okt 형태소 분석기 초기화
okt = Okt()

# CSV 파일 로드
file_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/tokenized_filtered_data_filled.csv"  # CSV 파일 경로
df = pd.read_csv(file_path)

# 제거할 품사 목록
unwanted_pos = set([
    'Punctuation',    # 문장부호
    'Foreign',        # 외국어
    'KoreanParticle', # 조사, 어미 등
    'Josa',           # 조사
    'Determiner',     # 관형사
    'Number',         # 숫자
    'Unknown',        # 알 수 없는 문자    
    'Adverb',         # 부사
    'Exclamatioin',   # 감탄사
    'PreEomi',        # 선어말어미
    'Eomi',           # 어미
    'Suffix',         # 접미사
    'Hashtag',        # 해쉬태그       
])

def pos_text_filter(text):
    try:
        # text가 이미 원문이면, 그냥 직접 토큰화
        pos_tags = okt.pos(text, norm=True, stem=True)
        filtered = [word for word, pos in pos_tags if pos not in unwanted_pos and len(word) > 1]
        return filtered
    except Exception as e:
        return []

# # pos 태깅 후 필터링 함수: 명사, 동사, 형용사만 필터링
# def pos_filter(tokens_str):
#     try:
#         tokens = ast.literal_eval(tokens_str)
#         pos_tags = okt.pos(" ".join(tokens))
#         filtered = [word for word, pos in pos_tags if pos not in unwanted_pos and len(word) > 1]
        # return filtered
#     except Exception as e:
#         return []

# tags 컬럼 처리 함수: 리스트로 변환하고, 각 태그를 품사 태깅 후 필터링
def parse_and_filter_tags(tag_string):
    try:
        tags_list = ast.literal_eval(tag_string)  # 문자열 -> 리스트
        if not isinstance(tags_list, list):
            return []
        # 각 태그를 띄어쓰기 기준으로 토큰화 및 품사 태깅
        all_filtered = []
        for tag in tags_list:
            pos_tags = okt.pos(tag)
            filtered = [word for word, pos in pos_tags if pos not in unwanted_pos and len(word) > 1]
            all_filtered.extend(filtered)
        return all_filtered
    except Exception:
        return []

# 컬럼별 토큰화
df["title_tokens"] = df["title"].apply(pos_text_filter)
df["tags_tokens"] = df["tags"].apply(parse_and_filter_tags)
df['img_text_tokens'] = df["img_text"].apply(pos_text_filter)

print(df[["title", "title_tokens"]].head(3))
print(df[["tags", "tags_tokens"]].head(3))
print(df[["img_text", "img_text_tokens"]].head(3))

# 결과를 새로운 CSV 파일로 저장
output_path = "C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/preprocessed_data_comp.csv"
df.to_csv(output_path, index=False)