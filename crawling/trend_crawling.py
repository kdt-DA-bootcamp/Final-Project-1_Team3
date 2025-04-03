import time
import pandas as pd
from playwright.sync_api import sync_playwright

def crawl_google_trends():
    url = "https://trends.google.co.kr/trending?geo=KR&hours=168"

    start_time = time.time()  # 실행 시간 측정

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        page.goto(url, timeout=60000)

        # 네트워크 로딩 완료까지 대기
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("div.mZ3RIc", timeout=15000)  # 메인 키워드 로딩 대기

        # 크롤링할 데이터 저장 리스트
        all_trends = []

        while True:
            previous_count = 0  # 스크롤 전 키워드 개수 저장

            for _ in range(10):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(5000)  # 데이터를 충분히 로드할 수 있도록 대기
                
                # 현재 키워드 개수 확인
                main_keywords = page.locator("div.mZ3RIc").all()
                if len(main_keywords) == previous_count:  # 더 이상 새로운 데이터가 로드되지 않으면 중지
                    break
                previous_count = len(main_keywords)  # 최신 키워드 개수 저장

            # 1. 메인 키워드 크롤링
            main_keyword_texts = [kw.inner_text().strip() for kw in main_keywords if kw.inner_text().strip()]

            # 2. 트렌드 분석 키워드 크롤링
            trend_analysis = page.locator("div.d15Ppf").all()
            trend_analysis_texts = [trend.inner_text().strip() for trend in trend_analysis if trend.inner_text().strip()]

            # 크롤링 데이터가 비어있다면, 크롤링 중단
            if not main_keyword_texts or not trend_analysis_texts:
                break

            # 데이터를 합쳐서 저장
            for i in range(min(len(main_keyword_texts), len(trend_analysis_texts))):
                keyword = main_keyword_texts[i] if i < len(main_keyword_texts) else "N/A"
                trend_related = trend_analysis_texts[i] if i < len(trend_analysis_texts) else "N/A"
                all_trends.append((keyword, trend_related))

            # 다음 페이지 버튼 찾기
            next_button = page.query_selector("button[aria-label='다음 페이지로 이동']")
            
            # 다음 페이지 버튼이 없거나 비활성화되어 있으면 종료
            if not next_button or not next_button.is_enabled():
                break  

            # 다음 페이지로 이동
            next_button.click()
            page.wait_for_load_state("networkidle")  # 페이지가 완전히 로드될 때까지 기다림
            page.wait_for_timeout(3000)  # 추가 대기

        browser.close()

    end_time = time.time()  # 실행 종료 시간 기록
    elapsed_time = end_time - start_time  # 실행 시간 계산
    print(f"크롤링 완료! 실행 시간: {elapsed_time:.2f}초")

    # CSV 저장
    df = pd.DataFrame(all_trends, columns=["main_keywords", "trend_keywords"])
    df.to_csv("trending_keywords.csv", index=False, encoding="utf-8-sig")
    print("CSV 파일 저장 완료: trending_keywords.csv")

    return all_trends

# 실행
trending_keywords = crawl_google_trends()