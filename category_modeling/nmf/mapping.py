import pandas as pd
import numpy as np
import ast
import pickle

# 1. 최종 카테고리 매핑 생성
final_cat_df = pd.read_csv("C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/modeling/nmf/최종_카테고리15.csv", encoding="utf-8-sig")

# 각 토픽 번호를 (categoryID, final_category) 튜플로 매핑하는 사전 생성
topic_to_category = {}
for idx, row in final_cat_df.iterrows():
    categoryID = row["categoryID"]
    final_cat = row["final_category"]
    topics_str = row["topics"]  # 예: "Topic #0, Topic #2"
    if pd.isnull(topics_str) or not isinstance(topics_str, str):
        topics_str = ""
    for token in topics_str.split(","):
        token = token.strip()
        if token.lower().startswith("topic"):
            token = token.replace("Topic", "").replace("#", "").strip()
        try:
            topic_num = int(token)
            topic_to_category[topic_num] = (categoryID, final_cat)
        except Exception as e:
            continue

print("최종 카테고리 매핑:")
print(topic_to_category)

# 2. preprocessed_data_comp.csv 파일 로드
data_df = pd.read_csv("preprocessed_data_comp.csv")
print("현재 데이터 컬럼:", data_df.columns.tolist())

# 3. combined_text 생성 함수
if "combined_text" not in data_df.columns:
    required_cols = ["title_tokens", "tags_tokens", "img_tokens"]
    existing_cols = [col for col in required_cols if col in data_df.columns]
    if existing_cols:
        def combine_tokens(row):
            tokens_list = []
            for col in existing_cols:
                if pd.notnull(row[col]):
                    try:
                        # 문자열로 저장된 리스트인 경우 ast.literal_eval로 변환
                        token_list = ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]
                        if isinstance(token_list, list):
                            tokens_list.extend(token_list)
                    except Exception as e:
                        pass
            return " ".join(tokens_list)
        data_df["combined_text"] = data_df.apply(combine_tokens, axis=1)
        print("combined_text 열을 생성하였습니다.")
    else:
        data_df["combined_text"] = ""
        print("토큰 관련 컬럼이 없어 combined_text 열을 빈 문자열로 생성하였습니다.")
else:
    print("combined_text 열이 이미 존재합니다.")

# 4. 저장된 TF-IDF 벡터라이저와 NMF 모델 로드
with open("C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/modeling/nmf/tfidf_vectorizer_15.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("C:/Users/hp/Desktop/Bootcamp/PROJECT_OTT_AARRR/modeling/nmf/nmf_model_15.pkl", "rb") as f:
    nmf_model = pickle.load(f)

# 5. 각 문서를 TF-IDF 벡터로 변환 후 NMF 모델로 토픽 분포 계산
docs = data_df["combined_text"].fillna("").tolist()
tfidf_matrix = tfidf_vectorizer.transform(docs)
W = nmf_model.transform(tfidf_matrix)  # 문서-토픽 행렬, shape=(num_docs, num_topics)

# 6. 각 문서별 주도 토픽(최대 가중치 토픽 번호) 결정
dominant_topics = np.argmax(W, axis=1)

# 7. 주도 토픽 번호에 해당하는 최종 카테고리와 categoryID 할당
final_category_list = []
categoryID_list = []
for t in dominant_topics:
    mapping = topic_to_category.get(t, ("Unknown", "Unknown"))
    categoryID, final_cat = mapping
    categoryID_list.append(categoryID)
    final_category_list.append(final_cat)

data_df["categoryID"] = categoryID_list
data_df["final_category"] = final_category_list

# 8. 결과 CSV 파일로 저장
output_file = "preprocessed_data_with_final_category_15.csv"
data_df.to_csv(output_file, index=False)
print(f"각 문서에 최종 카테고리와 categoryID가 부여되었습니다. 결과 파일: {output_file}")
