"""Streamlit explorer extending the award-winning team project.

The filename and the original reversal-point concept are retained while the
current screen adds role-specific and next-appearance views.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue import (
    add_pbp_context,
    build_audit_metrics,
    decile_summary,
    forward_decile_summary,
    load_dataset,
    load_pbp_features,
)


DATA_PATH = ROOT / "data" / "final" / "fatigue_with_index.csv"
PBP_PATH = ROOT / "data" / "external" / "pbp_appearance_2023_2024.csv"

st.set_page_config(page_title="KBO 투수 피로 신호 탐색", page_icon="⚾", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data
def get_pbp_data() -> pd.DataFrame:
    return load_pbp_features(PBP_PATH)


@st.cache_data
def get_metrics(frame: pd.DataFrame) -> dict:
    return build_audit_metrics(frame)


def observation_table(frame: pd.DataFrame, player: str, date: pd.Timestamp) -> pd.DataFrame:
    row = frame.loc[(frame["선수"] == player) & (frame["날짜"] == date)].iloc[0]
    values = {
        "선수": player,
        "등판일": row["날짜"].date().isoformat(),
        "팀 / 보직": f"{row['팀']} / {row['보직']}",
        "표본 내 점수 백분위": round(float(row["피로도지수_점수"]), 1),
        "투구수": int(row["투구수"]),
        "이닝": round(float(row["이닝"]), 1),
        "휴식일수": int(row["휴식일수"]),
        "WHIP": round(float(row["WHIP"]), 2),
        "FIP": round(float(row["FIP"]), 2),
        "GS": round(float(row["GS"]), 1),
    }
    return pd.DataFrame([values])


def process_table(frame: pd.DataFrame, player: str, date: pd.Timestamp) -> pd.DataFrame:
    row = frame.loc[(frame["선수"] == player) & (frame["날짜"] == date)].iloc[0]
    if pd.isna(row["pbp_csw_rate"]):
        return pd.DataFrame([{"안내": "2023–2024 투구 단위 매칭 자료가 없는 등판입니다."}])

    def difference(column: str, scale: float = 1.0) -> float | None:
        value = row[f"{column}_vs_prior5"]
        return None if pd.isna(value) else round(float(value) * scale, 1)

    values = {
        "강한 공 평균 구속": f"{row['pbp_hard_velocity']:.1f} km/h",
        "등판 후반 구속 변화": (
            "-" if pd.isna(row["pbp_late_velocity_delta"])
            else f"{row['pbp_late_velocity_delta']:+.1f} km/h"
        ),
        "최근 5회 대비 구속": (
            "-" if difference("pbp_hard_velocity") is None
            else f"{difference('pbp_hard_velocity'):+.1f} km/h"
        ),
        "강한 공 비중": f"{row['pbp_hard_usage'] * 100:.1f}%",
        "최근 5회 대비 강한 공": (
            "-" if difference("pbp_hard_usage", 100) is None
            else f"{difference('pbp_hard_usage', 100):+.1f}%p"
        ),
        "CSW%": f"{row['pbp_csw_rate'] * 100:.1f}%",
        "최근 5회 대비 CSW": (
            "-" if difference("pbp_csw_rate", 100) is None
            else f"{difference('pbp_csw_rate', 100):+.1f}%p"
        ),
        "존 통과율": f"{row['pbp_zone_rate'] * 100:.1f}%",
        "초구 스트라이크율": f"{row['pbp_first_pitch_strike_rate'] * 100:.1f}%",
        "7회 이후 2점차 이내 투구 비중": f"{row['pbp_high_pressure_share'] * 100:.1f}%",
    }
    return pd.DataFrame([values])


frame = add_pbp_context(get_data(), get_pbp_data())
metrics = get_metrics(frame)

st.title("⚾ KBO 투수 피로 신호 탐색")
st.caption("학술제 수상 분석의 개인 후속 고도화 대시보드 · 2020–2024 경기별 기록 · 17,528행 · 163명")
st.info(
    "원 프로젝트의 투수 교체 의사결정 지원 방향을 이어 받아, 저장된 점수와 경기 기록을 탐색하는 도구입니다. "
    "0–100 값은 전체 표본 내 백분위이며, 생리적 피로·부상 확률·교체 시점을 뜻하지 않습니다."
)

with st.sidebar:
    st.header("조회 조건")
    roles = st.multiselect("보직", sorted(frame["보직"].unique()), default=sorted(frame["보직"].unique()))
    years = st.multiselect("연도", sorted(frame["연도"].unique()), default=sorted(frame["연도"].unique()))

filtered = frame.loc[frame["보직"].isin(roles) & frame["연도"].isin(years)].copy()
if filtered.empty:
    st.warning("선택 조건에 해당하는 등판 기록이 없습니다.")
    st.stop()

players = sorted(filtered["선수"].unique())
left, right = st.columns(2)
with left:
    player_a = st.selectbox("투수 A", players, index=0, key="player_a")
with right:
    default_b = 1 if len(players) > 1 else 0
    player_b = st.selectbox("투수 B", players, index=default_b, key="player_b")

history = (
    filtered.loc[filtered["선수"].isin([player_a, player_b]), ["날짜", "선수", "피로도지수_점수"]]
    .pivot_table(index="날짜", columns="선수", values="피로도지수_점수", aggfunc="mean")
    .sort_index()
)

st.subheader("시간에 따른 표본 내 점수 백분위")
st.line_chart(history, y_label="백분위 점수", x_label="등판일", height=360)
st.caption(
    "서로 다른 날짜의 두 투수를 직접 비교하는 용도가 아니라, 각 선수의 기록 내에서 변화 시점을 찾는 탐색 화면입니다."
)

st.subheader("특정 등판 기록 비교")
col_a, col_b = st.columns(2)
with col_a:
    dates_a = filtered.loc[filtered["선수"] == player_a, "날짜"].sort_values(ascending=False).tolist()
    date_a = st.selectbox(
        f"{player_a} 등판일", dates_a,
        format_func=lambda value: value.date().isoformat(), key="date_a",
    )
    st.dataframe(observation_table(filtered, player_a, date_a), hide_index=True, use_container_width=True)
with col_b:
    dates_b = filtered.loc[filtered["선수"] == player_b, "날짜"].sort_values(ascending=False).tolist()
    date_b = st.selectbox(
        f"{player_b} 등판일", dates_b,
        format_func=lambda value: value.date().isoformat(), key="date_b",
    )
    st.dataframe(observation_table(filtered, player_b, date_b), hide_index=True, use_container_width=True)

st.caption("관측값을 나란히 제시할 뿐, 어느 선수를 기용해야 하는지 자동 권고하지 않습니다.")

with st.expander("투구 단위 프로세스 지표 · 2023–2024"):
    st.caption(
        "경기 결과뿐 아니라 구속 유지, 헛스윙·루킹 스트라이크(CSW), 존 통과율을 함께 봅니다. "
        "최근 5회 기준은 같은 선수·같은 보직의 이전 등판만 사용합니다."
    )
    st.caption(
        "데이터: [slothman3878/kbo_playbyplay](https://huggingface.co/datasets/"
        "slothman3878/kbo_playbyplay) · CC BY 4.0 · 2023–2024 파생 지표"
    )
    process_a, process_b = st.columns(2)
    with process_a:
        st.markdown(f"**{player_a} · {date_a.date().isoformat()}**")
        st.dataframe(
            process_table(filtered, player_a, date_a),
            hide_index=True, use_container_width=True,
        )
    with process_b:
        st.markdown(f"**{player_b} · {date_b.date().isoformat()}**")
        st.dataframe(
            process_table(filtered, player_b, date_b),
            hide_index=True, use_container_width=True,
        )

st.divider()
st.subheader("점수 구간과 경기 성과")
horizon = st.radio(
    "분석 시점", ["동일 경기", "다음 동일 보직 등판"], horizontal=True,
    help="후속 분석에서는 현재 점수가 다음 등판까지 이어지는지도 별도로 확인합니다.",
)
if horizon == "동일 경기":
    summary = decile_summary(filtered).set_index("score_decile")
else:
    summary = forward_decile_summary(frame, roles=roles, years=years).set_index("score_decile")
metric = st.selectbox("성과 지표", ["WHIP", "ERA", "FIP", "GS"])
st.line_chart(summary[[metric]], y_label=f"중앙값 {metric}", x_label="점수 10분위", height=320)
if horizon == "동일 경기":
    st.caption("학술제에서 제시한 피로도 구간과 같은 경기 성과의 관계를 재현한 화면입니다.")
else:
    st.caption(
        "현재 점수와 같은 선수의 다음 동일 보직 등판을 연결한 개인 후속 분석입니다. "
        "선발과 불펜의 운용 구조가 다르므로 보직 필터와 함께 해석해야 합니다."
    )

with st.expander("72.6 기준은 현재 화면에서 어떻게 활용하나요?"):
    st.write(
        "72.6은 학술제 분석에서 도출한 교체·휴식 검토 후보선이며, 현재 화면에서도 컨디션을 "
        "추가로 확인하는 기준으로 유지합니다. 저장된 데이터에서 회복실패 라벨 ROC AUC는 "
        f"{metrics['classification_diagnostics']['recovery_failure_auc']:.3f}, "
        "부상위험도 대리 라벨 AUC는 "
        f"{metrics['classification_diagnostics']['injury_proxy_auc']:.3f}입니다. "
        "이는 지수 하나만으로 회복·부상을 확률화하기보다 경기 기록과 함께 의사결정을 보조하는 "
        "방식이 적절하다는 후속 분석 결과입니다."
    )
    st.write(
        "후속 버전은 원래의 역전점 아이디어를 이어 받아 선발·불펜과 동일 경기·다음 등판을 "
        "나누어 보여 줍니다. 향후 실제 회복·부상 라벨과 외부 시즌이 연결되면 확률 보정과 "
        "비용 기반 자동 추천까지 단계적으로 확장할 수 있습니다."
    )
