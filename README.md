# KBO 투수 피로도 분석 — 역전점 모델링

KBO 선발 투수의 피로 누적이 성적에 영향을 주기 시작하는 시점(역전점)을
탐색하고, 누적 등판·이닝 기반의 피로도 지수를 구성해 시계열·SHAP 해석·
Streamlit 앱까지 한 흐름으로 연결한 데이터 분석 프로젝트.

## 진행 기간
2025 시즌 데이터 기반 — "최강이세용" 팀 프로젝트 (수상작, [docs/상장.jpg](./docs/상장.jpg))

## 역할
팀 프로젝트 — 데이터 수집(크롤링)·전처리·모델링·해석·Streamlit 배포 전 흐름 수행

## 데이터
| 데이터 | 출처 | 비고 |
|---|---|---|
| 선수 기록·구사율·평균구속 | 스탯티즈, 마이KBO 크롤링 | 2025 시즌 |
| 선수 바이오 | 자체 크롤러 | 신장·체중·연차 등 |
| 통합 데이터 | `data/processed/merged_*.csv` | 6개 머지 산출물 |
| 피로도 지수 최종본 | `data/final/fatigue_with_index.csv` | 모델링용 최종 |

## 분석 흐름

| 단계 | 노트북 | 내용 |
|---|---|---|
| 0 | `00_whip_보간.ipynb` | WHIP 결측 보간 |
| 1 | `01_데이터_모델링_1차.ipynb` | 1차 회귀·트리 모델 비교 |
| 2 | `02_역전점_구하기_2차_모델링.ipynb` | 피로 누적이 성적을 역전시키는 임계점 탐색 |
| 3 | `03_시계열.ipynb` | 등판 간격·이닝 누적의 시계열 분해 |
| 4 | `04_시각화.ipynb` | 피로도 지수 분포·역전점 분포 |
| 5 | `05_데이터_통합.ipynb` | 최종 통합 (`data/final/`) |

크롤링 단계 분리:
`crawlers/스탯티즈_크롤링.ipynb`, `crawlers/투수_크롤링.py`,
`crawlers/타자_크롤링.py`, `crawlers/바이오_크롤링.py`,
`crawlers/소속팀_크롤링.py`, `crawlers/데이터_병합.ipynb`

## 기술 스택
| 영역 | 도구 |
|---|---|
| 데이터 처리 | pandas, numpy, scipy |
| 크롤링 | selenium, undetected-chromedriver, beautifulsoup4, cloudscraper |
| 모델링 | scikit-learn, statsmodels, xgboost, lightgbm, catboost |
| 해석 | SHAP |
| 시각화 | matplotlib, seaborn |
| 앱 | Streamlit |

## 산출물
- 피로도 지수가 부착된 최종 데이터: [`data/final/fatigue_with_index.csv`](./data/final/fatigue_with_index.csv)
- 역전점 탐색 Streamlit 앱: [`app/역전점_앱.py`](./app/역전점_앱.py)
- 발표 자료: [`docs/최강이세용_발표_자료.pdf`](./docs/최강이세용_발표_자료.pdf)
- 프로젝트 요약서: [`docs/최강이세용_프로젝트_요약서.pdf`](./docs/최강이세용_프로젝트_요약서.pdf)
- 참고 문헌 6편: [`docs/references/`](./docs/references/) — 투수 피로도·MLB 부상 예측·KBO 연봉 모델링 등

## 디렉토리 구조
```
KBO-pitcher-fatigue/
├── README.md
├── LICENSE
├── requirements.txt
├── .devcontainer/              # GitHub Codespaces 환경
│
├── crawlers/                   # 스탯티즈·마이KBO·바이오 크롤링
├── data/
│   ├── raw/                    # mykbo 2025 원본 CSV
│   ├── processed/              # 머지 산출물 6개
│   └── final/                  # 피로도 지수 부착 최종
├── notebooks/                  # 분석 단계 노트북 00~05
├── app/                        # Streamlit 역전점 앱
│
└── docs/
    ├── references/             # 학술 참고문헌 6편 (PDF)
    ├── 최강이세용_발표_자료.pdf
    ├── 최강이세용_프로젝트_요약서.pdf
    ├── 최강이세용.docx
    └── 상장.jpg                # 수상 증명
```
