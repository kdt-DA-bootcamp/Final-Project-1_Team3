import platform
import matplotlib.pyplot as plt
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
    """Matplotlib 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: NanumGothic)"""
    if platform.system() == "Windows":
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == "Darwin":
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')

    matplotlib.rcParams['axes.unicode_minus'] = False