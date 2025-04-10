import pandas as pd

# CSV 파일 읽기 (파일 경로에 맞게 수정)
df = pd.read_csv('data/videos_by_keyword.csv')

# categoryID가 29 또는 30인 행 제거
df = df[~df['categoryID'].isin([29, 30])]

# categoryID에 따른 카테고리명 (초기 카테고리 매핑)
categories = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    19: "Travel & Events",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology"
}

# 각 세부 카테고리에 대한 그룹 매핑 (대분류: 그룹번호, 그룹이름)
group_mapping = {
    'Film & Animation': (1, '엔터테인먼트'),
    'Music': (1, '엔터테인먼트'),
    'Comedy': (1, '엔터테인먼트'),
    'Entertainment': (1, '엔터테인먼트'),
    'Autos & Vehicles': (2, '차량'),
    'Travel & Events': (3, '여행/음식'),
    'Gaming': (4, '게임'),
    'Sports': (5, '스포츠'),
    'People & Blogs': (6, '라이프'),
    'Howto & Style': (6, '라이프'),
    'News & Politics': (7, '정치'),
    'Pets & Animals': (8, '반려동물'),
    'Education': (9, '교육'),
    'Science & Technology': (10, '과학/기술')
}

# categoryID -> (top_categoryID, top_category) 매핑
id_to_group = {cid: group_mapping.get(cat_name, (cid, None)) for cid, cat_name in categories.items()}

# 각 행의 categoryID를 기반으로 그룹 정보를 할당하는 함수
def assign_top_category(cid):
    return id_to_group.get(cid, (cid, None))  # 기존 categoryID 유지

# 각 카테고리별로 (그룹번호, 그룹이름) 튜플을 반환받아 두 개의 컬럼에 할당
df['categoryID'], _ = zip(*df['categoryID'].map(assign_top_category))

# categoryID 컬럼을 int로 변환 (만약 누락된 값이 없다는 가정하에)
df['categoryID'] = df['categoryID'].astype(int)

# top_category 컬럼 드롭
df = df.drop(columns=['top_category'], errors='ignore')

# 결과 CSV 파일로 저장
df.to_csv('data/videos_by_keyword.csv', index=False)
print("CSV 파일에 대분류 컬럼이 추가되어 저장되었습니다.")