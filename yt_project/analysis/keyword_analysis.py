import os
import sys
import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm
import time
import torch
from sentence_transformers import SentenceTransformer

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from config.db_config import get_db_engine
    print("DB 설정 모듈 불러오기 성공")
except ImportError as e:
    print(f"DB 설정 모듈 불러오기 실패: {e}")
    sys.exit(1)

# GPU 설정 확인
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 중인 장치: {device}")
    return device

# 키워드 추출 함수
def extract_keywords(text, kw_model):
    keywords = kw_model.extract_keywords(text, top_n=5)
    return [kw[0] for kw in keywords]

# 데이터 불러오기
def load_data():
    try:
        engine = get_db_engine()
        if engine is None:
            print("DB 엔진 생성 실패")
            return pd.DataFrame()

        # 데이터 로드
        videos_df = pd.read_sql("SELECT * FROM video", engine)
        channels_df = pd.read_sql("SELECT * FROM channel", engine)
        videocategory_df = pd.read_sql("SELECT * FROM videocategory", engine)
        thumbnail_df = pd.read_sql("SELECT * FROM thumbnail", engine)

        # 컬럼 구조 확인
        print("Videos columns:", list(videos_df.columns))
        print("Channels columns:", list(channels_df.columns))
        print("VideoCategory columns:", list(videocategory_df.columns))
        print("Thumbnail columns:", list(thumbnail_df.columns))

        # 데이터 병합
        merged_df = pd.merge(videos_df, channels_df, on="channelID", how="left")
        merged_df = pd.merge(merged_df, videocategory_df, left_on="categoryID", right_on="id", how="left")
        merged_df = pd.merge(merged_df, thumbnail_df, on="thumbnailURL", how="left")

        # 병합 후 컬럼 확인
        print("Merged columns after join:", list(merged_df.columns))

        # 중복 컬럼 처리
        if 'title_x' in merged_df.columns and 'title_y' in merged_df.columns:
            merged_df['title'] = merged_df['title_x'].fillna('') + ' ' + merged_df['title_y'].fillna('')
            merged_df = merged_df.drop(columns=['title_x', 'title_y'])
        elif 'title_x' in merged_df.columns:
            merged_df['title'] = merged_df['title_x']
            merged_df = merged_df.drop(columns=['title_x'])
        elif 'title_y' in merged_df.columns:
            merged_df['title'] = merged_df['title_y']
            merged_df = merged_df.drop(columns=['title_y'])
        elif 'title' not in merged_df.columns:
            print("Warning: 'title' column not found after merging.")
            merged_df['title'] = ''  # 임시 조치

        # 최종 데이터 컬럼 선택
        selected_columns = ['videoID', 'title', 'tags', 'imgText', 'channelTitle', 
                            'subscriberCount', 'viewCount', 'likeCount', 'commentCount', 
                            'uploadDate', 'duration', 'categoryID']

        # 존재하지 않는 컬럼은 제외
        selected_columns = [col for col in selected_columns if col in merged_df.columns]
        merged_df = merged_df[selected_columns]

        # 키워드 컬럼이 없는 경우 추가
        if 'keywords' not in merged_df.columns:
            merged_df['keywords'] = ''

        print("데이터 불러오기 성공")
        print("최종 데이터 컬럼 확인:", list(merged_df.columns))
        return merged_df

    except Exception as e:
        print(f"데이터 불러오기 실패: {e}")
        return pd.DataFrame()

# 데이터 저장 경로 설정 함수
def get_save_path():
    # 프로젝트 루트 경로
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # AWS 환경일 경우
    if os.name == 'posix':
        # 프로젝트 루트 밑의 data 폴더로 설정
        return os.path.join(project_root, 'data', 'keyword_analysis.csv')
    # Windows 환경일 경우 (로컬 PC)
    elif os.name == 'nt':
        # 프로젝트 루트 밑의 data 폴더로 설정
        return os.path.join(project_root, 'data', 'keyword_analysis.csv')
    else:
        raise EnvironmentError("지원하지 않는 OS입니다.")

# 키워드 분석 및 저장
def analyze_keywords(save_every=500):
    device = get_device()
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    kw_model = KeyBERT(model=embedding_model)

    df = load_data()
    if df.empty:
        print("데이터 로드에 실패했습니다.")
        return

    start_time = time.time()
    total = len(df)
    print(f"총 {total}개 중 키워드 미추출 항목부터 시작합니다...")

    # 저장 경로 가져오기
    save_path = get_save_path()
    save_dir = os.path.dirname(save_path)

    # 디렉토리 존재 여부 확인 및 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(range(total), desc='키워드 추출 중'):
        text = f"{df.at[i, 'title']} {df.at[i, 'tags']} {df.at[i, 'imgText']}"
        keywords = extract_keywords(text, kw_model)

        # 키워드 리스트를 문자열로 변환하여 저장
        df.at[i, 'keywords'] = ', '.join(keywords)

        if i % save_every == 0 and i > 0:
            df.to_csv(save_path, index=False)
            elapsed = time.time() - start_time
            avg = elapsed / (i+1)
            remaining = (total - i - 1) * avg
            print(f"중간 저장: {i}/{total} 경과: {elapsed:.1f}s | 예상 남은 시간: {remaining/60:.1f}분")

    # 최종 저장
    df.to_csv(save_path, index=False)
    elapsed = time.time() - start_time
    print(f"키워드 추출 완료! 총 소요 시간: {elapsed/60:.2f}분")

# 메인 실행 부분
if __name__ == '__main__':
    analyze_keywords()