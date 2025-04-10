import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import matplotlib

def get_font_path():
    """WordCloud용 폰트 경로 반환 (Windows: malgun.ttf, 기타: None)"""
    if platform.system() == "Windows":
        return "C:/Windows/Fonts/malgun.ttf"
    elif platform.system() == "Darwin":
        return "/System/Library/Fonts/AppleGothic.ttf"
    else:
        return "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

def set_korean_font():
    if platform.system() == "Linux":
        font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'NanumGothic.ttf')
        
        print("🛠️ 폰트 경로 확인:", font_path)
        print("📂 해당 경로에 파일 있음?", os.path.exists(font_path))

        if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                print("✅ NanumGothic 로컬 로딩 완료:", font_prop.get_name())
        else:
                print("❌ NanumGothic.ttf 경로를 찾을 수 없음")
    elif platform.system() == "Windows":
            plt.rc('font', family='Malgun Gothic')
    elif platform.system() == "Darwin":
            plt.rc('font', family='AppleGothic')

    plt.rcParams['axes.unicode_minus'] = False