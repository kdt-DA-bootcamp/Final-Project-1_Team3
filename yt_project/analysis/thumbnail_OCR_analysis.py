# ------ 썸네일 텍스트 추천(영한님 작성) ------
#OCR 전처리
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import pymysql
import ast
import re
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import json
from langchain_openai import ChatOpenAI
from config.db_config import get_db_connection  # DB 연결 함수 재사용
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# .env 파일 로드 (필요 시)
load_dotenv()


def clean_text(text):
    """
    주어진 텍스트에 대해 간단한 전처리를 수행합니다.
    
    1. 소문자 변환
    2. 공백 기준 토큰화 후, 각 토큰에 영어 알파벳(a-z, A-Z) 또는 한글(가-힣)이 포함된 경우만 유지
    3. 유지한 토큰들을 공백으로 결합하여 반환
    """
    if pd.isnull(text):
        return ""
    
    text = text.lower()
    tokens = text.split()
    cleaned_tokens = [token for token in tokens if re.search('[a-zA-Z가-힣]', token)]
    return " ".join(cleaned_tokens)


def load_and_clean_thumbnail_texts() -> pd.DataFrame:
    """
    DB에서 video와 thumbnail 데이터를 불러와, 
      - imgText 컬럼의 결측치를 'unknown'으로 채우고,
      - thumbnailURL 컬럼 결측치 여부를 확인한 후,
      - imgText 컬럼에 대해 clean_text() 함수를 적용하여 'cleaned_Texts' 컬럼을 생성한 DataFrame을 반환합니다.
    
    또한, 결과 DataFrame을 아래 지정된 경로에 pickle 파일로 저장합니다.
    
    Returns:
        pd.DataFrame: 전처리된 DataFrame
    """
    # DB 연결 생성 (db_config.py의 get_db_connection 함수 사용)
    conn = get_db_connection()
    if conn is None:
        raise ConnectionError("DB 연결에 실패했습니다.")
    
    cursor = conn.cursor()
    query = """
        SELECT v.thumbnailURL, t.imgText, v.likeCount, v.viewCount, v.commentCount, v.categoryID, v.videoID
        FROM video v
        JOIN thumbnail t ON v.thumbnailURL = t.thumbnailURL;
    """
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()

    # DataFrame 생성 및 기본 전처리
    df = pd.DataFrame(data)
    df['imgText'] = df['imgText'].fillna('unknown')

    if df['thumbnailURL'].isnull().sum() > 0:
        print("thumbnailURL 열에 결측치가 존재합니다. 추가 전처리가 필요합니다.")
    else:
        print("thumbnailURL 열에 결측치가 없습니다.")

    # imgText 컬럼 전처리: clean_text() 함수 적용
    df['cleaned_Texts'] = df['imgText'].apply(clean_text)

    # # 결과 저장: 지정된 경로에 pickle 파일로 저장 (경로는 코드 내에서 지정)
    # save_path =  "data/thumbnail_Texts_cleaned.pkl"
    # df.to_pickle(save_path)
    # print(f"전처리된 데이터가 .pkl 파일로 저장되었습니다: {save_path}")

    return df


if __name__ == "__main__":
    load_and_clean_thumbnail_texts()


# 임베딩

# --- KoBERT 모델 및 토크나이저 초기화 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model_name = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

def process_text(text: str) -> str:
    """
    문자열 리스트 형식의 텍스트를 파이썬 리스트로 변환한 뒤,
    리스트이면 공백으로 결합하여 하나의 문장으로 반환합니다.
    변환 실패 시 원본 텍스트를 반환합니다.
    """
    try:
        tokens = ast.literal_eval(text)
        if isinstance(tokens, list):
            return " ".join(tokens)
        else:
            return text
    except Exception:
        return text

def get_kobert_embedding(text: str):
    """
    입력된 텍스트에 대해 KoBERT 임베딩 벡터를 추출합니다.
    (여기서는 CLS 토큰의 임베딩을 반환하며, numpy 배열 형태로 리턴합니다.)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding[0].cpu().numpy()

def extract_kobert_embeddings(df: pd.DataFrame, text_col: str = "cleaned_Texts", output_path: str = None) -> pd.DataFrame:
    """
    주어진 DataFrame의 text_col 컬럼에 대해 아래 처리를 진행합니다.
      1. process_text()를 적용해 'processed_text' 컬럼 생성
      2. get_kobert_embedding()을 적용해 'embedding' 컬럼 생성
      3. 결과 DataFrame 반환 및 (output_path가 지정된 경우) pickle 파일로 저장

    Parameters:
        df (pd.DataFrame): 입력 데이터 (필수 컬럼: text_col, videoID, categoryID 등)
        text_col (str): 전처리할 텍스트 컬럼 이름 (기본 'cleaned_Texts')
        output_path (str): 저장할 pickle 파일 경로 (지정하지 않으면 저장하지 않음)

    Returns:
        pd.DataFrame: 'processed_text'와 'embedding' 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    df['processed_text'] = df[text_col].apply(process_text)
    df['embedding'] = df['processed_text'].apply(get_kobert_embedding)
    
    print("임베딩 결과 확인:")
    print(df[['videoID', 'categoryID', 'processed_text', 'embedding']].head())
    
    if output_path:
        df.to_pickle(output_path)
        print(f"전처리된 데이터가 .pkl 파일로 저장되었습니다: {output_path}")
    
    return df

if __name__ == "__main__":
    # 모듈 테스트용 예시: 임시 DataFrame 생성 후 임베딩 수행 및 저장
    sample_data = {
        "videoID": [1, 2],
        "categoryID": [10, 20],
        "cleaned_Texts": ["['안녕하세요', '반갑습니다']", "['테스트', '문장']"]
    }
    df_sample = pd.DataFrame(sample_data)
    output_path = "data/thumbnail_Texts_embedding.pkl"
    extract_kobert_embeddings(df_sample, text_col="cleaned_Texts", output_path=output_path)


# 데이터 셋 형성
def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    주어진 DataFrame으로부터 입력 피처와 타겟 변수를 구성하고, 학습 및 테스트 데이터로 분리합니다.
    
    DataFrame은 반드시 다음 컬럼들을 포함해야 합니다:
      - 'videoID'
      - 'categoryID'
      - 'viewCount'
      - 'likeCount'
      - 'commentCount'
      - 'embedding' : 각 행에 numpy 배열 형태의 임베딩 벡터가 저장되어 있음.
    
    처리 단계:
      1. 카테고리 전처리: One-Hot Encoding
      2. OCR 임베딩 벡터 준비: 각 행의 embedding 컬럼에서 2차원 배열 생성
      3. 입력 피처 구성: embedding과 One-Hot 인코딩된 카테고리 결합 (column-wise)
      4. 출력(Target) 변수 구성: viewCount, likeCount, commentCount를 추출하고 로그 변환 적용 (np.log1p)
      5. train/test 분리: 기본 80/20 비율
      
    Parameters:
        df (pd.DataFrame): 입력 DataFrame
        test_size (float): 테스트 셋의 비율 (기본 0.2)
        random_state (int): 데이터 분할 시 랜덤 시드 (기본 42)
        
    Returns:
        dict: {
            "X_train": 학습용 입력 피처,
            "X_test": 테스트용 입력 피처,
            "y_train": 학습용 타겟 (로그 변환 적용),
            "y_test": 테스트용 타겟 (로그 변환 적용),
            "encoder": OneHotEncoder 객체 (카테고리 인코딩 결과 복원 등 추가 용도)
        }
    """
    # 1. 카테고리 전처리: One-Hot Encoding
    encoder = OneHotEncoder(sparse=False)  # 최신 버전에서는 sparse_output=False를 사용해도 됨
    category_encoded = encoder.fit_transform(df[['categoryID']])
    print("One-Hot Encoded Category Shape:", category_encoded.shape)
    
    # 2. embedding 벡터 준비: 각 행의 embedding 컬럼을 2차원 배열로 변환
    # embedding 컬럼은 이미 numpy 배열(예: shape=(embedding_dim,))를 포함하고 있어야 합니다.
    embedding_matrix = np.vstack(df['embedding'].values)
    print("Embedding Matrix Shape:", embedding_matrix.shape)
    
    # 3. 입력 피처 구성: 임베딩과 One-Hot 인코딩된 카테고리 결합 (column-wise)
    X = np.concatenate([embedding_matrix, category_encoded], axis=1)
    print("Final Input Feature Shape:", X.shape)
    
    # 4. 출력(Target) 변수 구성: viewCount, likeCount, commentCount 열 추출 및 로그 변환
    y = df[['viewCount', 'likeCount', 'commentCount']].values
    y_log = np.log1p(y)
    
    # 5. 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=test_size, random_state=random_state)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoder": encoder
    }

# 모듈 테스트 (직접 실행할 때만 동작)
if __name__ == "__main__":
    # 예시: 임의 DataFrame 생성 (실제 사용 시 DB 또는 파일에서 df를 불러올 것)
    sample_data = {
        'videoID': [1, 2, 3, 4],
        'categoryID': [10, 20, 10, 30],
        'viewCount': [100, 200, 150, 300],
        'likeCount': [10, 20, 15, 30],
        'commentCount': [1, 2, 1, 3],
        'embedding': [np.random.rand(768) for _ in range(4)]
    }
    df_sample = pd.DataFrame(sample_data)
    
    dataset = prepare_dataset(df_sample)


# 모델 학습
def train_popularity_model(X_train, y_train, X_test, y_test, model_save_path=None):
    """
    XGBoost 회귀 모델을 MultiOutputRegressor로 감싸서 학습 및 평가한 후, 
    지정된 경로에 모델을 저장합니다.
    
    Parameters:
        X_train (np.ndarray): 학습 입력 피처
        y_train (np.ndarray): 학습 타겟 (로그 변환 적용한 값)
        X_test (np.ndarray): 테스트 입력 피처
        y_test (np.ndarray): 테스트 타겟 (로그 변환 적용한 값)
        model_save_path (str): 모델 저장 경로 (예: "data/thumbnail_Texts_model.pkl")
        
    Returns:
        model: 학습 완료된 MultiOutputRegressor 모델
    """
    # 1. 모델 정의 및 학습
    popularity_model = MultiOutputRegressor(
        xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    )
    popularity_model.fit(X_train, y_train)
    print("모델 학습 완료!")
    
    # 2. 예측 수행 및 평가 지표 계산
    y_pred = popularity_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("평가 결과:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² Score:", r2)
    
    # 3. 모델 저장 (선택 사항)
    if model_save_path:
        with open(model_save_path, 'wb') as f:
            pickle.dump(popularity_model, f)
        print("모델 저장 완료!")
    
    return popularity_model


def recommend_thumbnail_text(candidate_text, category, popularity_model, encoder, get_kobert_embedding):
    """
    후보 썸네일 문구(candidate_text)와 카테고리 정보를 활용하여 인기도 예측 결과를 산출하고,
    추천 점수 및 예측 인기도 벡터를 반환합니다.
    
    Parameters:
        candidate_text (str): 후보 썸네일 문구
        category (str or int): 해당 문구의 카테고리 (DB와 일치하는 형식; 내부에서는 문자열로 변환)
        popularity_model: 학습된 회귀 모델 (MultiOutputRegressor)
        encoder: OneHotEncoder 객체 (카테고리 인코딩에 사용)
        get_kobert_embedding: 함수, 입력 텍스트에 대해 KoBERT 임베딩을 추출
        
    Returns:
        score (float): 추천 점수
        predicted_popularity (np.ndarray): 예측 인기도 벡터 (shape: (1, 3))
    """
    # 1. KoBERT 임베딩: candidate_text를 임베딩 벡터로 변환
    candidate_embedding = get_kobert_embedding(candidate_text)  # shape: (embedding_dim,)
    
    # 2. 카테고리 One-Hot 인코딩 (입력은 문자열 형태)
    category_encoded = encoder.transform([[str(category)]])
    
    # 3. 입력 피처 구성: candidate_embedding과 category_encoded 결합
    candidate_embedding = candidate_embedding.reshape(1, -1)
    candidate_feature = np.concatenate([candidate_embedding, category_encoded], axis=1)
    
    # 4. 인기도 예측
    predicted_popularity = popularity_model.predict(candidate_feature)  # shape: (1, 3)
    
    # 5. 추천 점수 산출: 가중치 적용 (예: [0.5, 0.3, 0.2])
    weights = np.array([0.5, 0.3, 0.2])
    score = (predicted_popularity * weights).sum()
    
    return score, predicted_popularity


def recommend_keywords_for_category(category, candidate_texts, popularity_model, encoder, get_kobert_embedding, top_n=5):
    """
    특정 카테고리 내 후보 썸네일 문구들을 대상으로 추천을 진행하여,
    상위 top_n 개의 추천 결과 (문구와 추천 점수)를 반환합니다.
    
    Parameters:
        category (str or int): 추천할 카테고리 (문자열 또는 숫자; 내부에서는 문자열 처리)
        candidate_texts (list of str): 해당 카테고리에 속하는 후보 썸네일 문구 리스트
        popularity_model: 학습된 회귀 모델
        encoder: OneHotEncoder 객체
        get_kobert_embedding: 함수, 텍스트 임베딩 추출용
        top_n (int): 상위 추천 문구 개수 (기본 5)
    
    Returns:
        list of tuples: [(문구, 추천 점수), ...] 상위 top_n 결과
    """
    recommendations = []
    for text in candidate_texts:
        score, _ = recommend_thumbnail_text(text, str(category), popularity_model, encoder, get_kobert_embedding)
        recommendations.append((text, score))
        
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


def generate_recommendations(df, popularity_model, encoder, get_kobert_embedding, top_n=5):
    """
    DataFrame의 각 카테고리별로 추천 결과를 생성합니다.
    
    Parameters:
        df (pd.DataFrame): 입력 데이터 (반드시 'categoryID'와 'processed_text' 컬럼 포함)
        popularity_model: 학습된 회귀 모델
        encoder: OneHotEncoder 객체
        get_kobert_embedding: KoBERT 임베딩을 추출하는 함수
        top_n (int): 각 카테고리별 상위 추천 결과 수 (기본 5)
    
    Returns:
        dict: {카테고리: [{"text": 문구, "score": 추천점수}, ...], ...}
    """
    recommendations_dict = {}
    unique_categories = df['categoryID'].unique()
    
    for cat in unique_categories:
        candidate_texts = df[df['categoryID'] == cat]['processed_text'].tolist()
        recommendations = recommend_keywords_for_category(str(cat), candidate_texts, popularity_model, encoder, get_kobert_embedding, top_n=top_n)
        recommendations_dict[str(cat)] = [{"text": text, "score": score} for text, score in recommendations]
    
    return recommendations_dict


def save_recommendations_json(recommendations_dict, output_path):
    """
    추천 결과 딕셔너리를 JSON 파일로 저장합니다. (UTF-8 인코딩)
    
    Parameters:
        recommendations_dict (dict): 추천 결과 딕셔너리
        output_path (str): JSON 파일 저장 경로
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendations_dict, f, ensure_ascii=False, indent=4)
    print(f"추천 결과가 {output_path} 파일에 저장되었습니다.")


#LLM 후처리


# 환경 변수 로드
load_dotenv()

# 프로젝트 설정 관련 함수
def project_setup(name: str) -> None:
    """
    프로젝트 이름을 기반으로 LangSmith 관련 환경 변수를 설정하고 추적을 시작합니다.
    """
    os.environ['LANGSMITH_PROJECT'] = name
    os.environ['LANGSMITH_TRACING'] = 'true'
    print(f"프로젝트: {name} 추적 시작")


def stop_tracing() -> None:
    """
    LangSmith 추적 중지
    """
    os.environ['LANGSMITH_TRACING'] = 'false'
    print("LangSmith 추적 중지")


# LLM 관련 설정
OPENAI_MODELS = {
    "gpt-4o": {
        'price': {
            'input': 2.50,
            'cached': 1.25,
            'output': 10.00,
        }
    },
    "gpt-4o-mini": {  # 가장 저렴한 모델
        'price': {
            'input': 0.15,
            'cached': 0.075,
            'output': 0.60,
        }
    },
    "o1": {
        'price': {
            'input': 15.00,
            'cached': 7.50,
            'output': 60.00,
        }
    },
    "o1-mini": {
        'price': {
            'input': 1.10,
            'cached': 0.55,
            'output': 4.40,
        }
    },
    "o3-mini": {  # 최신 모델 예시
        'price': {
            'input': 1.10,
            'cached': 0.55,
            'output': 4.40,
        }
    },
}
MODEL_NAMES = list(OPENAI_MODELS.keys())



def create_llm(model_name: str = "gpt-4o", temperature: float = 0.1, max_tokens: int = 2048) -> ChatOpenAI:
    """
    LangChain의 ChatOpenAI 객체를 생성합니다.
    
    Parameters:
        model_name (str): 사용할 모델명 (기본 "gpt-4o")
        temperature (float): 창의성 조절 (기본 0.1)
        max_tokens (int): 최대 토큰 수 (기본 2048)
    
    Returns:
        ChatOpenAI 객체
    """
    llm = ChatOpenAI(
        temperature=temperature,
        max_tokens=max_tokens,
        model_name=model_name,
    )
    return llm


def refine_recommendations(loaded_recommendations: dict, llm: ChatOpenAI) -> dict:
    """
    각 카테고리별 후보 추천 문구를 LLM으로 후처리하여 다듬습니다.
    
    Parameters:
        loaded_recommendations (dict): { category: [ {"text": 후보문구}, ... ], ... }
        llm (ChatOpenAI): LangChain LLM 객체
        
    Returns:
        dict: { category: [ {"original_text": 원본, "refined_text": 다듬은문구}, ... ], ... }
    """
    refined_recommendations = {}

    # 각 카테고리별로 반복
    for category, candidate_list in loaded_recommendations.items():
        refined_recommendations[category] = []
        if not candidate_list:
            continue

        for rec in candidate_list:
            candidate_text = rec["text"]
            prompt = f"""아래의 유튜브 썸네일 문구 후보를 더욱 자연스럽고 매력적인 문구로 다듬어 주세요.

후보 문구: "{candidate_text}"

최종 문구는 한국어로 작성해 주세요. 아래처럼 "최종 문구:"는 넣지 마세요.
예시:
원본: coca-co 이제 더 없어!!!!
변환: "코카콜라, 이제는 끝났어요!"
"""
            # LLM 호출하여 결과 문자열 획득 (content 속성 사용)
            refined_caption = llm.invoke(prompt).content
            refined_recommendations[category].append({
                "original_text": candidate_text,
                "refined_text": refined_caption
            })
    return refined_recommendations


def save_refined_recommendations(refined_recommendations: dict, output_path: str) -> None:
    """
    추천 결과 딕셔너리를 pickle 파일로 저장합니다.
    
    Parameters:
        refined_recommendations (dict): 후처리된 추천 결과 딕셔너리
        output_path (str): 저장할 파일 경로
    """
    with open(output_path, "wb") as f:
        pickle.dump(refined_recommendations, f)
    print(f"추천 결과가 {output_path} 파일에 저장되었습니다.")


def load_refined_recommendations(pkl_path: str) -> dict:
    """
    pickle 파일에서 추천 결과 딕셔너리를 불러옵니다.
    
    Parameters:
        pkl_path (str): pickle 파일 경로
        
    Returns:
        dict: 추천 결과 딕셔너리
    """
    with open(pkl_path, "rb") as f:
        refined_data = pickle.load(f)
    return refined_data


def print_refined_recommendations(refined_data: dict) -> None:
    """
    추천 결과 딕셔너리의 각 카테고리별 추천 문구를 출력합니다.
    
    Parameters:
        refined_data (dict): 추천 결과 딕셔너리
    """
    for category, recommendations in refined_data.items():
        print(f"\n{'='*10} 카테고리 {category}의 추천 문구 {'='*10}\n")
        for idx, item in enumerate(recommendations, 1):
            original_text = item["original_text"]
            refined_text = item["refined_text"]
            print(f"{idx}. 원본: {original_text}")
            print(f"   변환: {refined_text}\n")


def main():
    # 1. OCR 전처리: DB에서 데이터를 불러와 imgText 전처리 수행
    print("===== Step 1: OCR 전처리 시작 =====")
    df_text = load_and_clean_thumbnail_texts()
    # df_text에는 'thumbnailURL', 'imgText', 'cleaned_Texts', 등 필요한 컬럼이 포함됨

    # 2. KoBERT 임베딩 추출: cleaned_Texts 컬럼에 대해 임베딩 계산 (저장 경로는 내부에 지정)
    print("===== Step 2: KoBERT 임베딩 추출 시작 =====")
    df_emb = extract_kobert_embeddings(df_text, text_col="cleaned_Texts", 
                                        output_path="data/thumbnail_Texts_embedding.pkl")
    
    # 3. 데이터 셋 형성: embedding, categoryID, viewCount, likeCount, commentCount 등을 활용하여 train/test 데이터 구성
    print("===== Step 3: 데이터 셋 형성 =====")
    dataset = prepare_dataset(df_emb, test_size=0.2, random_state=42)
    # dataset에는 X_train, X_test, y_train, y_test, encoder 등이 포함됨

    # 4. 인기도 회귀 모델 학습: XGBoost 기반 멀티 출력 회귀 모델 학습 및 저장
    print("===== Step 4: 인기도 회귀 모델 학습 =====")
    popularity_model = train_popularity_model(
        dataset["X_train"],
        dataset["y_train"],
        dataset["X_test"],
        dataset["y_test"],
        model_save_path="data/thumbnail_Texts_model.pkl"
    )

    # 5. 추천 생성: 전처리된 df_emb를 바탕으로 각 카테고리별 추천 결과 생성
    print("===== Step 5: 추천 생성 =====")
    encoder = dataset["encoder"]
    recommendations_dict = generate_recommendations(df_emb, popularity_model, encoder, get_kobert_embedding, top_n=5)
    save_recommendations_json(recommendations_dict, "data/recommendations.json")
    
    # 6. LLM 후처리: 추천된 문구들을 LLM (LangChain)으로 다듬기
    print("===== Step 6: LLM 후처리 시작 =====")
    project_setup("LangChain Test")  # 프로젝트명과 LangSmith 추적 관련 환경 변수 설정
    llm = create_llm(model_name="gpt-4o", temperature=0.1, max_tokens=2048)
    refined_recs = refine_recommendations(recommendations_dict, llm)
    save_refined_recommendations(refined_recs, "data/refined_recommendations.pkl")
    
    # 결과 확인
    print("===== 최종 후처리된 추천 결과 =====")
    for category, recs in refined_recs.items():
        print(f"카테고리 {category}:")
        for rec in recs:
            print(f"원본: {rec['original_text']}  ->  변환: {rec['refined_text']}")
    

if __name__ == "__main__":
    main()

