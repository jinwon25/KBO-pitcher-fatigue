# 검증 대기 신규 시즌 스냅샷

이 디렉터리는 정본·앱·학습·외부 holdout 입력이 아닙니다. 2020–2024 정본은 그대로 유지합니다.

## 실제 수집 결과 (2026-10-07)

출처: [slothman3878/kbo_playbyplay](https://huggingface.co/datasets/slothman3878/kbo_playbyplay),
revision `6afc8af044e3bba5f326b688e8cb41d7ff7065ec`, CC BY 4.0.
NAVER Sports/KBO 기록에서 만든 구조화 파생 데이터입니다. 원문 중계는 포함하지 않으며,
원 출처의 권리·약관은 데이터셋 라이선스와 별도입니다. 이번 CSV는 해당 원본에서 재집계한 파생물입니다.

| 구분 | 2025 | 2026 부분 자료 |
| --- | ---: | ---: |
| 수록 기간 | 3월 22일–10월 4일 | 3월 28일–7월 26일 |
| 경기 | 720 | 470 |
| 경기별 투수 등판 | 7,025 | 4,594 |
| 투수 ID | 281 | 275 |
| 실제 투구 행 | 217,848 | 144,728 |
| 무투구 사건 | 122 | 51 |
| 타석 | 56,139 | 37,192 |
| 선수별 같은 날짜 추가 등판 | 23 | 0 |
| 일정 API의 완료 경기 | 720 | 709 (10월 7일까지) |
| 그중 PBP 미수록 경기 | 0 | 239 |
| PBP 마지막 날짜까지 미수록 완료 경기 | 0 | 0 |

일정 API는 NAVER의 **비공식·문서화되지 않은 엔드포인트**입니다. 공식 KBO 개방 API나
독립된 공식 기록 인증이 아닙니다. 완료 경기의 원천 ID·날짜·홈/원정 팀을 대조했고,
취소 84경기(2025), 73경기(2026 조회 구간)는 완료 경기에서 제외했습니다.
2026의 `schedule_pending_games=0`은 **10월 7일까지 조회 구간**에만 해당하며 시즌 종료를 뜻하지 않습니다.

각 연도 폴더의 `public_pbp_v1` 파일:

- `.csv`: `(PlayerID, GameID)`별 등판 과정 지표. 날짜나 투구수로 다른 출처와 자동 결합하지 않습니다.
- `.re24.csv`: 해당 스냅샷의 24개 주자·아웃 기대득점. 기술 통계이며 holdout 학습에 사용하면 안 됩니다.
- `.schedule.csv`: 조회 구간 전체 일정의 최소 필드.
- `.schedule.json.gz`: 실제 API 응답과 요청 URL을 보존한 압축 스냅샷. 오프라인 재현 입력입니다.
- `.manifest.json`: 원본 revision·SHA-256, 라이선스, 집계 보존 검사, 일정 누락 목록, 산출물 해시.

`season_complete=false`, `holdout_ready=false`입니다. 2025의
`schedule_completed_coverage=true`는 조회 일정과의 경기 집합 일치만 뜻합니다.
`ScheduledStartKST`는 예정 시작 시각이며 실제 첫 투구 시각으로 간주하지 않습니다.

## 재현

설치: `pip install -r requirements-dev.txt`. 원본 parquet는 `.cache/kbo_playbyplay/`에
해시를 검증해 저장합니다. 아래 명령은 새로운 파일명으로만 게시합니다.

```bash
python scripts/backfill_public_pbp.py --year 2025 --download \
  --through 2025-12-31 \
  --schedule data/staging/2025/public_pbp_v1.schedule.json.gz \
  --output data/staging/2025/public_pbp_rebuilt.csv

python scripts/backfill_public_pbp.py --year 2026 --download \
  --through 2026-10-07 \
  --schedule data/staging/2026/public_pbp_v1.schedule.json.gz \
  --output data/staging/2026/public_pbp_rebuilt.csv
```

원본이 캐시에 있으면 `--download` 없이 네트워크 접근 없이 실행할 수 있습니다.
최신 일정 재수집은 `--collect-schedule --schedule .cache/<새 파일명>.json`을 지정합니다.
월별 조회 누락·중복 경기·실패 응답·잘린 응답은 오류로 처리합니다. API는 큰 조회에서
`gameTotalCount`도 상한값으로 반환하므로 `len(games)==gameTotalCount`만으로 완결을 판단하지 않습니다.

2026 PBP는 고정 revision의 7월 26일 자료입니다. 일정만 새로 조회해도 투구 데이터는 늘지 않습니다.
후속 공개 원본을 확보하면 `backfill_public_pbp.py`의 해당 연도 `SOURCES` revision·해시를
검토·갱신하고 별도 캐시와 **새 스냅샷 이름**으로 재실행해야 합니다. 기존 2023–2024 CLI의
고정 revision이나 이전 연도 설정은 변경하지 않습니다.
API 중계 전체를 대량 수집하는 기능이나 자동 정본 승격은 포함하지 않습니다.

## 남은 검증과 적용 순서

1. **공식 기록 대조:** 팀별 경기 수, 선수별 등판·투구수·상대 타자 합계를 독립된 공식 기록과 대조합니다.
   PBP 게임 누락이 없어도 모든 사건·선수 기록이 정확하다는 보장은 없습니다.
2. **출처 간 crosswalk:** MYKBO 경기/선수 ID ↔ PBP ID를 원천 링크와 박스스코어로 검증합니다.
   기존 2025 중복 13그룹 모두 새 PBP에서 두 경기와 투구수 다중집합이 일치합니다.
   그러나 김영우 5월 17일은 두 경기 모두 15구여서 투구수도 식별 키가 될 수 없습니다.
   기존 26행에는 ID를 추정해 쓰지 않았습니다.
3. **부가자료·라벨:** 날짜 단위 구속·구사율·날씨와 부상/엔트리 관측 기간을 재검토하고
   경기별로 확보합니다. 새 과정 지표만으로 기존 피로도/부상 라벨을 복원하지 않습니다.
4. **분석 계약 전환:** 실제 경기 순서·보직·같은 날 휴식 간격·다음 등판 정의를 정하고
   앱/학습 경로를 경기 단위로 이행합니다. RE24는 훈련 기간에서만 추정해 holdout에 적용합니다.
5. **2026 후속 백필:** 시즌 종료와 최종 재편성/취소/서스펜디드 처리를 확인한 뒤 최신 원본으로
   같은 검증을 반복합니다. 이후 별도 검토를 거쳐 외부 시즌 검증에 편입합니다.

현재 수집 범위와 검증 결과는 스냅샷에 고정되어 있습니다. 향후 실행을 예약한 것은 아닙니다.
