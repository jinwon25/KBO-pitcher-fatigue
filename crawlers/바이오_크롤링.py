import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import re

# 1) cloudscraper 세션 생성 (Cloudflare 우회)
scraper = cloudscraper.create_scraper()

players = {
    "조동욱": 2585,
    "한승혁": 327,
    "황준서": 2588,
    "정우주": 2762,
    "주현상": 1125,
    "김범수": 990,
    "김종수": 204,
    "김기중": 2127,
    "김서현": 2433,
    "박상원": 1548,

    "백승현": 1043,
    "함덕주": 62,
    "임찬규": 372,
    "장현식": 572,
    "김진성": 529,
    "김영우": 2773,
    "이지강": 1902,
    "박명근": 2450,
    "손주영": 1501,
    "송승기": 2215,
    "이정용": 1827,
    "유영찬": 2049,

    "배찬승": 2756,
    "최원태": 1096,
    "김대호": 2649,
    "김재윤": 1155,
    "김태훈": 462,
    "이호성": 2431,
    "이승현_우": 670,
    "이승현_좌": 2122,
    "이승민": 1956,
    "양창섭": 1651,
    "육선엽": 2581,

    "최준용": 1951,
    "홍민기": 1952,
    "정철원": 1638,
    "정현수": 2576,
    "김강현": 943,
    "김원중": 1241,
    "이민석": 2262,
    "나균안": 1506,

    "조상우": 512,
    "최지민": 2287,
    "전상현": 1374,
    "정해영": 1967,
    "김대유": 461,
    "김현수": 1790,
    "이형범": 567,
    "이준영": 1028,
    "네일": 2590,
    "성영탁": 2674,

    "배제성": 950,
    "주권": 1211,
    "김재원": 2787,
    "김민수": 1147,
    "고영표": 1142,
    "이상동": 1858,
    "박영현": 2302,
    "소형준": 1990,
    "원상현": 2610,
    "우규민": 356,

    "앤더슨": 2729,
    "최민준": 1676,
    "정동윤": 1341,
    "전영준": 2399,
    "조병현": 2151,
    "김광현": 396,
    "김민": 1696,
    "김택형": 1078,
    "이로운": 2481,
    "노경은": 17,
    "박시후": 2057,

    "로건": 2782,
    "배재환": 633,
    "최성영": 1316,
    "전사민": 1851,
    "김진호": 1514,
    "김영규": 1686,
    "이준혁": 2372,
    "류진욱": 1103,
    "신민혁": 1761,

    "곽빈": 1628,
    "어빈": 2748,
    "김택연": 2567,
    "이영하": 1346,
    "로그": 2745,
    "박치국": 1473,
    "박정수": 1017,
    "박신지": 1632,

    "알칸타라": 1856,
    "조영건": 1843,
    "주승우": 2311,
    "정현우": 2799,
    "김선기": 1678,
    "이준우": 2884,
    "오석주": 1577,
    "박정훈": 2793,
    "박윤성": 2545,
    "원종현": 549,
    "윤석원": 2390
}

rows = []
for name, pid in players.items():
    url = f"https://mykbostats.com/players/{pid}"
    html = scraper.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    # height
    h_el = soup.select_one('span[itemprop="height"]')
    height = h_el.text.strip() if h_el else ""

    # weight
    w_el = soup.select_one('span[itemprop="weight"]')
    weight = w_el.text.strip() if w_el else ""

    # birthYear
    b_el = soup.select_one('span[itemprop="birthDate"]')
    by = ""
    if b_el:
        m = re.search(r"\b(\d{4})\b", b_el.text)
        by = m.group(1) if m else ""

    rows.append([name, height, weight, by])

df = pd.DataFrame(rows, columns=["PlayerName","Height","Weight","BirthYear"])
df.to_csv("mykbo_bio_core.csv", index=False, encoding="utf-8-sig")
print("✅ Saved mykbo_bio_core.csv")
