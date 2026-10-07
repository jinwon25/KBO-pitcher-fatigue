# 수집 코드 안내

이 폴더는 학술제 당시 MYKBO·스탯티즈·KBO 웹페이지에서 경기·선수·구종·구속 자료를 수집한 코드와 병합 노트북을 보존합니다.

웹사이트 구조와 브라우저 드라이버는 시간이 지나며 변경되므로 이 코드는 수집 당시의 작업 근거로 제공합니다. 재수집 시에는 선수 목록과 셀렉터를 현재 사이트에 맞게 갱신해야 합니다.

현재 저장소의 자동 재현 경로는 정본 `data/final/fatigue_with_index.csv`에서 시작하며, 수집 코드는 원천 데이터 구축 과정을 보여 주는 자료입니다.

## 2025 이후 경기 식별 수집·병합

`투수_크롤링.py`는 선수 페이지 ID와 **각 날짜 셀의 경기 링크**를 함께 저장합니다.
`/games/13023`과 `/games/13023-KT-vs-LG-20250517`은 동일한
`mykbo:game:13023`으로 정규화합니다. 등판 키는 `(PlayerID, GameID)`입니다.
이름·날짜·팀·경기 시작 시각·행 순서로 ID를 만들지 않습니다.
동명이인도 원천 선수 ID가 다르면 서로 다른 선수로 취급합니다.

```bash
# players.json: {"박영현": "2302", "장현식": "572"}처럼 원천 ID를 지정
python crawlers/투수_크롤링.py --players players.json --year 2025 --snapshot pitchers-v1
# 브라우저에서 해당 연도를 선택. 각 행의 날짜가 연도와 다르면 저장 중단.
python scripts/stage_season.py \
  --input data/staging/2025/pitchers-v1.csv \
  --year 2025 --output data/staging/2025/validated-v1.csv
```

수집에는 `requirements-legacy.txt`의 Selenium·undetected-chromedriver와 Chrome이 필요합니다.
선택한 시즌의 모든 행이 표시됐는지 확인해야 합니다. 헤더 변경, 경기 링크 누락,
stale DOM, 중복 키는 건너뛰지 않고 실패합니다. 현재 사이트 전체 수집에 대한
라이브 검증은 별도이며, 로컬 회귀 테스트는 원천 확인 fixture로 수행합니다.

`--input`은 여러 선수 파일을 받을 수 있고 `--features`에는 같은 등판 집합의
구속·구종 등 **ID를 보존한** CSV를 지정할 수 있습니다. 서로 다른 출처의 숫자 ID는
같은 경기라는 뜻이 아닙니다. 스탯티즈/KBO/PBP 자료를 MYKBO와 결합하려면
경기 및 선수 ID를 모두 검증한 crosswalk가 필요합니다. 날짜만 가진 구사율·날씨
파일을 등판별 자료로 간주해 자동 결합하지 않습니다.

`데이터_병합.ipynb`의 날짜 merge와 `drop_duplicates`는 보존용 기록이며
신규 백필 경로가 아닙니다. 안전한 경로는
[`stage_season.py`](../scripts/stage_season.py)의 일대일 결합입니다.
상세 검증·완결 조건은 [백필 안내](../docs/SEASON_BACKFILL.md)를 참고하세요.
