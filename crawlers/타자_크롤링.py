import time
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

# 1) undetected-chromedriver로 드라이버 실행
driver = uc.Chrome()
wait = WebDriverWait(driver, 10)

# 2) 크롤링할 타자 선수명→페이지ID 매핑
players = {
    "김성진": 1899,
    "최원영": 2367,
    "전경원": 1674
}
years = ["2024", "2023", "2022", "2021", "2020"]
all_data = []

for name, pid in players.items():
    driver.get(f"https://mykbostats.com/players/{pid}")
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.ui.dropdown")))

    for year in years:
        resp = input(f"▶ [{name}] {year} 데이터가 없으면 '없음'을, 아니면 엔터… ")
        if resp.strip() == "없음":
            print(f"  - [{name}] {year} 스킵")
            continue

        time.sleep(0.5)  # 렌더 안정화

        # 5) 반드시 텍스트가 Show More인 버튼만 클릭
        while True:
            try:
                btns = driver.find_elements(
                    By.XPATH,
                    "//a[@phx-click='show_all' and normalize-space(text())='Show More']"
                )
                if not btns:
                    break
                # 여러 개 뜰 수 있으니 전부 JS 클릭
                for btn in btns:
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.2)
            except StaleElementReferenceException:
                time.sleep(0.2)
                continue

        # 6) 테이블 스크래핑 (타자 18컬럼)
        rows = driver.find_elements(By.CSS_SELECTOR, "table.sortable tbody tr")
        year_rows = []
        for row in rows:
            try:
                cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
            except StaleElementReferenceException:
                continue
            if len(cols) == 18:
                year_rows.append(cols)

        # 7) 없으면 “없음” 한 줄, 아니면 실제 데이터
        if not year_rows:
            all_data.append([name, year] + ["없음"] * 18)
        else:
            for cols in year_rows:
                all_data.append([name, year] + cols)

driver.quit()

# 8) DataFrame 생성 및 저장
columns = [
    "PlayerName","Year","Date","Opp","AB","R","H","2B","3B",
    "HR","RBI","BB","HBP","BA","OBP","SLG","OPS","SB","CS","GDP"
]
df = pd.DataFrame(all_data, columns=columns)
df.to_csv("mykbo_batters_by_name.csv", index=False, encoding="utf-8-sig")

print("✅ Saved mykbo_batters_by_name.csv")