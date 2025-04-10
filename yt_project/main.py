# 모든 파트 작성 후 최종 임포트 조정 필요
import argparse
import sys
from analysis.success_video_analysis import analyze as success_analyze, save_results as save_success_results
from analysis.thumbnail_analysis import main as thumbnail_analyze, save_color_data
from analysis.keyword_analysis import analyze as keyword_analyze, save_results as save_keyword_results

def main():
    parser = argparse.ArgumentParser(description="분석 실행 및 결과 저장")
    parser.add_argument('--analyze', action='store_true', help="분석을 실행하여 결과를 저장합니다.")
    args = parser.parse_args()
    
    if args.analyze:
        print("분석을 시작합니다...")

        # 키워드 분석
        try:
            keyword_results = keyword_analyze()
            save_keyword_results(keyword_results, "data/keyword_analysis_results.pkl")
            print("[키워드 분석] 완료 및 저장")
        except Exception as e:
            print(f"[키워드 분석] 오류 발생: {e}")

        # 썸네일 분석
        try:
            thumbnail_results = thumbnail_analyze()
            save_color_data(thumbnail_results, "data/thumbnail_analysis_results.csv")
            print("[썸네일 분석] 완료 및 저장")
        except Exception as e:
            print(f"[썸네일 분석] 오류 발생: {e}")

        # 반짝 영상 분석
        try:
            success_results = success_analyze()
            save_success_results(success_results, "data/success_video_results.pkl")
            print("[반짝 영상 분석] 완료 및 저장")
        except Exception as e:
            print(f"[반짝 영상 분석] 오류 발생: {e}")

        print("모든 분석 완료.")
    else:
        print("분석 옵션이 선택되지 않았습니다. --analyze 옵션을 사용하세요.")
        sys.exit(1)

if __name__ == '__main__':
    main()
