import time
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException

# undetected-chromedriver로 드라이버 실행
driver = uc.Chrome()
wait = WebDriverWait(driver, 10)

# 크롤링할 투수 선수명→페이지ID 매핑
players = {

}

years = ["2025"]
all_data = []

for name, pid in players.items():
    driver.get(f"https://mykbostats.com/players/{pid}")
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.ui.dropdown")))

    for year in years:
        resp = input(f"▶ [{name}] {year} 데이터가 없으면 '없음'을, 아니면 엔터를 눌러주세요… ")
        if resp.strip() == "없음":
            print(f"  - [{name}] {year} 스킵")
            continue

        time.sleep(0.5)

        # Show More 클릭
        while True:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, 'a[phx-click="show_all"]')
                btn.click()
                time.sleep(0.2)
            except (NoSuchElementException, TimeoutException):
                break
            except StaleElementReferenceException:
                time.sleep(0.2)
                continue

        # 테이블 스크래핑
        rows = driver.find_elements(By.CSS_SELECTOR, "table.sortable tbody tr")

        # ← 여기를 추가
        year_rows = []
        for row in rows:
            try:
                tds = row.find_elements(By.TAG_NAME, "td")
                cols = [td.text.strip() for td in tds]
            except StaleElementReferenceException:
                continue

            if len(cols) == 16:
                year_rows.append(cols)

        # 데이터가 없으면 "없음" 한 줄, 있으면 실제 데이터 모두 추가
        if not year_rows:
            all_data.append([name, year] + ["없음"] * 16)
        else:
            for cols in year_rows:
                all_data.append([name, year] + cols)

# 드라이버 종료
driver.quit()

# DataFrame 생성 및 저장
columns = [
    "PlayerName", "Year", "Date", "Opp", "Role", "Dec", "ERA", "WHIP",
    "IP", "NP", "R", "ER", "H", "HR", "SO", "BB", "HB", "GS"
]
df = pd.DataFrame(all_data, columns=columns)
df.to_csv("mykbo_pitchers_by_name.csv", index=False, encoding="utf-8-sig")

print("✅ Saved mykbo_pitchers_by_name.csv")