"""Follow-up Streamlit explorer for the award-winning team project.

The filename is retained for compatibility with the original project link.
The app itself no longer fits a leaky live model or issues lineup decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue import build_audit_metrics, decile_summary, load_dataset


DATA_PATH = ROOT / "data" / "final" / "fatigue_with_index.csv"

st.set_page_config(page_title="KBO 투수 피로 신호 탐색", page_icon="⚾", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


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


frame = get_data()
metrics = get_metrics(frame)

st.title("⚾ KBO 투수 피로 신호 탐색")
st.caption("학술제 수상 분석의 개인 후속 검증 대시보드 · 2020–2024 경기별 기록 · 17,528행 · 163명")
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

st.divider()
st.subheader("점수 구간과 같은 경기 성과")
summary = decile_summary(filtered).set_index("score_decile")
metric = st.selectbox("성과 지표", ["WHIP", "ERA", "FIP", "GS"])
st.line_chart(summary[[metric]], y_label=f"중앙값 {metric}", x_label="점수 10분위", height=320)
st.caption(
    "같은 경기의 기술적 연관입니다. 점수 개발에 경기력 관련 변수가 사용되어 예측력이나 인과효과로 해석할 수 없습니다."
)

with st.expander("왜 현재 앱은 72.6 자동 추천을 보류하나요?"):
    st.write(
        "저장된 데이터로 재계산하면 회복실패 라벨 ROC AUC는 "
        f"{metrics['classification_diagnostics']['recovery_failure_auc']:.3f}, "
        "부상위험도 비결측 여부로 만든 과거 대리 라벨 AUC는 "
        f"{metrics['classification_diagnostics']['injury_proxy_auc']:.3f}입니다. "
        "두 값 모두 무작위 기준 0.5에 가까워 현재 데이터만으로 운영 임계값을 뒷받침하기 어렵습니다."
    )
    st.write(
        "학술제 당시 앱은 WHIP를 포함한 선수기량 지수로 WHIP를 다시 예측해 목표 누수가 있었고, "
        "문서의 GS 예측 설명과도 달랐습니다. 원 아이디어와 결과는 기록으로 보존하되, 현재 앱은 "
        "외부 시즌과 실제 라벨로 재검증하기 전까지 해당 자동 모델과 처방 문구를 사용하지 않습니다."
    )
