# ------ 썸네일 이미지 추천(영한님 작성) ------
#전처리
from config.db_config import get_db_engine
from config.db_config import get_db_connection
import pandas as pd
import numpy as np
import pickle
import ast
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from collections import Counter


# 기본 불용어
DEFAULT_STOPWORDS = {'youtube', 'top', 'video', 'meter'}

def clean_tokenized_keywords(token_str):
    """문자열 형태의 키워드를 토큰화하고 전처리합니다."""
    if pd.isnull(token_str):
        return []
    token_str = token_str.strip()
    try:
        if token_str.startswith('[') and token_str.endswith(']'):
            tokens = ast.literal_eval(token_str)
        else:
            tokens = token_str.split(',')
    except Exception:
        tokens = token_str.split(',')

    cleaned_tokens = []
    for token in tokens:
        token = token.strip("[]'\", ")
        token = re.sub(r'[^\w\s]', '', token)
        token = token.strip()
        if token:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def filter_stopwords(tokens, stopwords):
    """불용어 필터링"""
    return [t for t in tokens if t not in stopwords]

def load_and_clean_thumbnail_keywords(stopwords=None) -> pd.DataFrame:
    """
    DB에서 video 및 thumbnail 데이터를 로드하고 imgKw 컬럼을 전처리한 DataFrame을 반환합니다.

    Parameters:
        stopwords (set): 불용어 집합 (기본값은 내부 설정 사용)

    Returns:
        pd.DataFrame: 전처리된 DataFrame
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    # DB 연결
    conn = get_db_connection()
    if conn is None:
        raise ConnectionError("DB 연결에 실패했습니다.")

    cursor = conn.cursor()
    query = """
        SELECT v.thumbnailURL, t.imgKw, v.likeCount, v.viewCount, v.commentCount, v.categoryID, v.videoID
        FROM video v
        JOIN thumbnail t ON v.thumbnailURL = t.thumbnailURL;
    """
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()

    # DataFrame 생성 및 전처리
    df = pd.DataFrame(data)
    df['imgKw'] = df['imgKw'].fillna('unknown')

    if df['thumbnailURL'].isnull().sum() > 0:
        print("thumbnailURL 열에 결측치가 존재합니다. 추가 전처리가 필요합니다.")
    else:
        print("thumbnailURL 열에 결측치가 없습니다.")

    df['cleaned_keywords'] = df['imgKw'].apply(clean_tokenized_keywords)
    df['cleaned_keywords'] = df['cleaned_keywords'].apply(lambda tokens: filter_stopwords(tokens, stopwords))

    return df

# 임베딩


# 모델 및 장치 설정
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("사용 장치:", device)

def parse_token_list(token_value):
    if isinstance(token_value, (list, np.ndarray)):
        return list(token_value)
    if pd.isnull(token_value):
        return []
    try:
        tokens = ast.literal_eval(token_value)
        return tokens if isinstance(tokens, list) else []
    except Exception:
        return token_value.split()

def get_bert_embedding_keywords(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding[0].cpu().numpy()

def extract_bert_keyword_embeddings(
    df: pd.DataFrame,
    keyword_col: str = "cleaned_keywords",
    output_path: str = None
) -> pd.DataFrame:
    """
    BERT 임베딩 추출 후 'bert_keyword_vector' 컬럼으로 저장, 선택적으로 .pkl 저장.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        keyword_col (str): 키워드 컬럼명 (기본 'cleaned_keywords')
        output_path (str): 결과 저장 경로 (.pkl) – 지정 시 저장됨

    Returns:
        pd.DataFrame: 임베딩 결과가 포함된 DataFrame
    """
    all_embeddings = []

    for _, row in df.iterrows():
        keywords = parse_token_list(row[keyword_col])
        if not keywords:
            embedding = np.zeros(model.config.hidden_size)
        else:
            combined = " ".join(keywords)
            embedding = get_bert_embedding_keywords(combined)
        all_embeddings.append(embedding)

    df['bert_keyword_vector'] = all_embeddings

    if output_path:
        df.to_pickle(output_path)
        print(f"BERT 임베딩 결과가 저장되었습니다: {output_path}")

    print("임베딩 완료. 행렬 크기:", np.vstack(all_embeddings).shape)
    return df



# 데이터셋 형성


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_categories):
        super(MultiTaskModel, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_categories)
        )

    def forward(self, x):
        pred_reg = self.regressor(x)
        pred_class = self.classifier(x)
        return pred_reg, pred_class


def preprocess_dataset(data: pd.DataFrame, pca_dim: int = 512):
    """
    PCA 차원 축소 및 타겟 벡터 분리, train/test split 수행
    """
    print("데이터 전처리 중...")

    # 임베딩 벡터 → numpy 행렬
    X_keywords = np.vstack(data['bert_keyword_vector'].values)
    print("원본 임베딩 크기:", X_keywords.shape)

    # PCA 축소
    pca = PCA(n_components=pca_dim)
    X_keywords_reduced = pca.fit_transform(X_keywords)
    print("PCA 축소 후:", X_keywords_reduced.shape)

    # 회귀 타겟
    y_view = data['viewCount'].values
    y_like = data['likeCount'].values
    y_comment = data['commentCount'].values
    y_reg = np.stack([y_view, y_like, y_comment], axis=1)

    # 분류 타겟
    data['categoryID'] = data['categoryID'].astype(int)
    encoder = LabelEncoder()
    data['categoryID_encoded'] = encoder.fit_transform(data['categoryID'])
    y_class = data['categoryID_encoded'].values

    # video ID 보존
    video_ids = data['videoID'].values

    # split
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test, vid_train, vid_test = train_test_split(
        X_keywords_reduced, y_reg, y_class, video_ids, test_size=0.2, random_state=42
    )

    print("학습 데이터 크기:", X_train.shape)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_class_train": y_class_train,
        "y_class_test": y_class_test,
        "vid_train": vid_train,
        "vid_test": vid_test,
        "num_categories": len(np.unique(y_class))
    }


def train_model(X_train, y_reg_train, y_class_train, num_categories, input_dim=512, hidden_dim=256, num_epochs=20):
    """
    멀티태스크 회귀 + 분류 모델 학습
    """
    model = MultiTaskModel(input_dim, hidden_dim, num_categories)
    criterion_reg = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # TensorDataset
    X_tensor = torch.FloatTensor(X_train)
    y_reg_tensor = torch.FloatTensor(y_reg_train)
    y_class_tensor = torch.LongTensor(y_class_train)
    train_loader = DataLoader(TensorDataset(X_tensor, y_reg_tensor, y_class_tensor), batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y_reg, batch_y_class in train_loader:
            optimizer.zero_grad()
            pred_reg, pred_class = model(batch_X)
            loss_reg = criterion_reg(pred_reg, batch_y_reg)
            loss_class = criterion_class(pred_class, batch_y_class)
            loss = loss_reg + loss_class
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


def save_dataset_split(data: pd.DataFrame, dataset: dict, save_path: str):
    """
    전체 학습용 데이터를 pickle로 저장
    """
    df_copy = data.copy()
    df_copy.to_pickle(save_path)
    print(f"데이터셋 저장 완료: {save_path}")
    return save_path

#모델 학습



class KeywordRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KeywordRecommender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


def build_vocab_from_keywords(cleaned_keywords, vocab_size=500):
    """가장 자주 등장한 키워드 기반 vocab 생성"""
    all_keywords = []
    for kw_list in cleaned_keywords:
        if isinstance(kw_list, list):
            all_keywords.extend(kw_list)
        elif isinstance(kw_list, str):
            all_keywords.extend(kw_list.split())
    counter = Counter(all_keywords)
    return [kw for kw, _ in counter.most_common(vocab_size)]


def create_target_vector(kw_list, vocab):
    """동영상 단위 multi-hot 타겟 벡터 생성"""
    target = np.zeros(len(vocab), dtype=np.float32)
    tokens = kw_list if isinstance(kw_list, list) else kw_list.split()
    for word in tokens:
        if word in vocab:
            idx = vocab.index(word)
            target[idx] = 1.0
    return target


def prepare_keyword_dataset(data: pd.DataFrame, embedding_col='bert_keyword_vector', keyword_col='cleaned_keywords', vocab_size=500):
    """임베딩 벡터와 multi-hot 타겟으로 구성된 학습용 데이터셋 준비"""
    print("데이터셋 준비 중...")

    X = np.vstack(data[embedding_col].dropna().values)
    print("X shape:", X.shape)

    vocab = build_vocab_from_keywords(data[keyword_col], vocab_size)
    V = len(vocab)

    y_keywords = np.array([
        create_target_vector(kw_list, vocab) if isinstance(kw_list, (list, str))
        else np.zeros(V, dtype=np.float32)
        for kw_list in data[keyword_col]
    ])
    print("y_keywords shape:", y_keywords.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y_keywords, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=32, shuffle=False)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vocab": vocab,
        "input_dim": X.shape[1],
        "output_dim": V
    }


def train_keyword_model(train_loader, input_dim, hidden_dim, output_dim, num_epochs=20):
    """키워드 추천 모델 학습"""
    model = KeywordRecommender(input_dim, hidden_dim, output_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model


def save_keyword_model(model, vocab, input_dim, hidden_dim, output_dim, save_path: str):
    """모델 저장"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim
    }, save_path)
    print(f"모델 저장 완료: {save_path}")


# 최종 결과
class KeywordRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KeywordRecommender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


def load_keyword_model(model_path: str):
    """저장된 모델과 관련 정보 로드"""
    checkpoint = torch.load(model_path)
    model = KeywordRecommender(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("모델 및 관련 정보 불러오기 완료!")

    return model, checkpoint['vocab']


def recommend_keywords_by_category(model, data: pd.DataFrame, vocab: list, top_k: int = 20) -> dict:
    """
    각 categoryID에 대해 평균 임베딩 기반 키워드 추천

    Returns:
        dict: {categoryID: [(keyword, score), ...]}
    """
    category_recommendations = {}
    category_ids = data['categoryID'].unique()

    for cat_id in category_ids:
        subset = data[data['categoryID'] == cat_id].copy()
        subset = subset.dropna(subset=['bert_keyword_vector'])

        if len(subset) == 0:
            continue

        X_cat = np.vstack(subset['bert_keyword_vector'].values)
        X_tensor = torch.FloatTensor(X_cat)

        with torch.no_grad():
            preds = model(X_tensor)
            preds = torch.sigmoid(preds).cpu().numpy()

        avg_preds = np.mean(preds, axis=0)
        keyword_scores = list(zip(vocab, avg_preds))
        top_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_k]
        category_recommendations[cat_id] = top_keywords

    return category_recommendations


def save_recommendations(recommendations: dict, output_path: str):
    """추천 결과를 pickle 파일로 저장"""
    with open(output_path, 'wb') as f:
        pickle.dump(recommendations, f)
    print(f"추천 결과 저장 완료: {output_path}")


def main():
    # 1. 썸네일 키워드 정제
    df = load_and_clean_thumbnail_keywords()

    # 2. BERT 임베딩 + 저장
    df = extract_bert_keyword_embeddings(df, output_path="data/IMC_imbedding.pkl")

    # 3. 멀티태스크 학습용 데이터 생성 + 모델 학습 + 저장
    dataset = preprocess_dataset(df)
    multitask_model = train_model(
        dataset["X_train"],
        dataset["y_reg_train"],
        dataset["y_class_train"],
        num_categories=dataset["num_categories"]
    )
    save_dataset_split(df, dataset, "data/merged_data_set_0404.pkl")

    # 4. 키워드 예측용 학습 데이터 생성 + 모델 학습 + 저장
    keyword_dataset = prepare_keyword_dataset(df)
    keyword_model = train_keyword_model(
        train_loader=keyword_dataset["train_loader"],
        input_dim=keyword_dataset["input_dim"],
        hidden_dim=256,
        output_dim=keyword_dataset["output_dim"]
    )
    save_keyword_model(
        model=keyword_model,
        vocab=keyword_dataset["vocab"],
        input_dim=keyword_dataset["input_dim"],
        hidden_dim=256,
        output_dim=keyword_dataset["output_dim"],
        save_path="data/keyword_recommender_full.pth"
    )

    # 5. 키워드 추천 실행 및 저장
    loaded_model, vocab = load_keyword_model("data/keyword_recommender_full.pth")
    keyword_recs = recommend_keywords_by_category(loaded_model, df, vocab, top_k=20)
    save_recommendations(keyword_recs, "data/category_recommendations0407.pkl")


if __name__ == "__main__":
    main()