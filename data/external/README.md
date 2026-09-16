# 외부 공개 데이터

## KBO 투구 단위 파생 데이터

`pbp_appearance_2023_2024.csv`는 공개 데이터셋
[`slothman3878/kbo_playbyplay`](https://huggingface.co/datasets/slothman3878/kbo_playbyplay)의
2023·2024 정규시즌 투구 단위 자료를 현재 프로젝트의 `(선수, 날짜, 팀)`과 연결해 만든 등판 단위 파생 데이터입니다. 팀을 결합 키에 포함해 동명이인 투수의 기록이 합쳐지지 않도록 했습니다.

`re24_matrix_2023_2024.csv`는 같은 원천에서 생성한 24행 기대득점표입니다. `base_state_label`은 1·2·3루 점유를 나타내며, 예를 들어 `1-3`은 1루와 3루에 주자가 있는 상태입니다.

- 원 데이터 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 원 데이터 제작자: `slothman3878`
- 고정 revision: `6afc8af044e3bba5f326b688e8cb41d7ff7065ec`
- 사용 시즌: 2023, 2024
- 파생 데이터: 7,924등판, 128명
- 현재 정본 대비 매칭률: 97.45%
- 양쪽 투구수 상관: 0.9995

원본 parquet는 약 35MB이며 저장소에 중복 배포하지 않습니다. 다음 명령은 고정된 revision을 다운로드하고 SHA-256을 검사한 뒤 파생 CSV를 다시 생성합니다.

```bash
python scripts/build_pbp_features.py --download
```

원본 파일의 SHA-256은 다음과 같습니다.

| 파일 | SHA-256 |
|---|---|
| `kbo_pbp_2023.parquet` | `818f6016655b02fe48b8118281d1b04bfe3548d376fdc70131a41ea539341edb` |
| `kbo_pbp_2024.parquet` | `8332cd716cf0126a4ab0bf390383f43deff22ab320a57fb70d02b31025bdf553` |

## 파생 지표

| 컬럼 | 정의 |
|---|---|
| `pbp_hard_velocity` | 직구·투심·커터 평균 구속 |
| `pbp_hard_usage` | 전체 투구 중 직구·투심·커터 비중 |
| `pbp_csw_rate` | 루킹 스트라이크·헛스윙·번트 헛스윙 비중 |
| `pbp_zone_rate` | 타자별 스트라이크존을 통과한 투구 비중 |
| `pbp_first_pitch_strike_rate` | 타석 첫 구가 스트라이크 또는 인플레이인 비중 |
| `pbp_late_velocity_delta` | 등판 후반 1/3과 초반 1/3의 강한 공 평균 구속 차이 |
| `pbp_high_pressure_share` | 7회 이후 2점차 이내 상황에서 던진 투구 비중 |
| `pbp_entry_inning`, `pbp_entry_outs`, `pbp_entry_runners` | 등판 첫 투구 시점의 이닝·아웃·주자 수 |
| `pbp_entry_run_margin` | 등판 진입 시 투수 소속팀 관점의 점수 차(리드는 양수) |
| `pbp_close_late_entry` | 7회 이후, 2점차 이내에서 등판했는지 여부 |
| `pbp_re24_allowed` | 타석별 `실점 + RE_after − RE_before`의 등판 합계 |
| `pbp_re24_allowed_per_bf` | RE24 허용을 상대 타자 수로 나눈 값 |

`pbp_late_velocity_delta`는 강한 공이 6구 이상인 등판에만 계산합니다. 음수이면 등판 후반 구속이 초반보다 낮았다는 뜻입니다.

RE24 기대득점표는 2023–2024 전체 타석의 24개 주자·아웃 상태에서 이닝 종료까지의 평균 득점으로 계산합니다. 타석 도중 투수가 바뀐 97건은 타석을 끝낸 투수에게 해당 타석의 RE24를 귀속했습니다. 이 값은 현재 경기 환경의 득점 기대값이며, 승리확률을 사용하는 공식 gmLI나 WPA는 아닙니다.

원 데이터는 NAVER Sports 문자중계에서 파생된 비공식 자료입니다. 원 데이터 카드의 사용 고지에 따라 KBO 또는 NAVER의 공식 승인 자료로 표현하지 않으며, 원문 중계 텍스트는 포함하지 않습니다.
