"""Core, testable analysis functions.

The canonical dataset contains an original model score (`피로도지수`) and its
empirical percentile rank (`피로도지수_점수`). This module audits those stored
values; it does not claim that they diagnose physiological fatigue or determine
a safe pitcher substitution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "선수", "연도", "팀", "보직", "날짜", "투구수", "휴식일수",
    "FIP", "WHIP", "ERA", "GS", "피로도지수", "피로도지수_점수",
    "회복실패여부", "부상위험도",
}
OUTCOME_COLUMNS = ("FIP", "WHIP", "ERA", "GS")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the canonical CSV and apply deterministic typing and ordering."""
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")
    frame = frame.copy()
    frame["날짜"] = pd.to_datetime(frame["날짜"], errors="raise")
    frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)
    return frame.sort_values(["선수", "날짜"], kind="stable").reset_index(drop=True)


def validate_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return reproducibility checks and fail fast on broken invariants."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")

    duplicate_keys = int(frame.duplicated(["선수", "날짜"]).sum())
    expected_percentile = frame["피로도지수"].rank(method="average", pct=True) * 100
    percentile_max_error = float(
        np.abs(expected_percentile - frame["피로도지수_점수"]).max()
    )
    finite_scores = bool(
        np.isfinite(frame[["피로도지수", "피로도지수_점수"]].to_numpy()).all()
    )
    checks = {
        "rows": int(len(frame)),
        "players": int(frame["선수"].nunique()),
        "date_min": frame["날짜"].min().date().isoformat(),
        "date_max": frame["날짜"].max().date().isoformat(),
        "years": sorted(int(year) for year in frame["연도"].unique()),
        "duplicate_player_dates": duplicate_keys,
        "score_min": float(frame["피로도지수_점수"].min()),
        "score_max": float(frame["피로도지수_점수"].max()),
        "percentile_rank_max_error": percentile_max_error,
        "finite_scores": finite_scores,
        "injury_risk_missing_rate": float(frame["부상위험도"].isna().mean()),
    }
    if duplicate_keys:
        raise ValueError(f"(선수, 날짜) 중복 키가 {duplicate_keys}개 있습니다.")
    if not finite_scores:
        raise ValueError("피로도 점수에 무한대 또는 비수치 값이 있습니다.")
    if percentile_max_error > 1e-9:
        raise ValueError("피로도지수_점수가 원점수의 백분위 순위와 일치하지 않습니다.")
    return checks


def correlation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare row-level and within-player descriptive correlations."""
    rows: list[dict[str, Any]] = []
    centered_score = frame["피로도지수"] - frame.groupby("선수")["피로도지수"].transform("mean")
    for metric in OUTCOME_COLUMNS:
        centered_metric = frame[metric] - frame.groupby("선수")[metric].transform("mean")
        rows.append({
            "metric": metric,
            "overall_r": float(frame["피로도지수"].corr(frame[metric])),
            "within_player_r": float(centered_score.corr(centered_metric)),
        })
    return pd.DataFrame(rows)


def binary_auc(target: pd.Series, score: pd.Series) -> float:
    """Compute ROC AUC from ranks without a machine-learning dependency."""
    values = pd.DataFrame({"target": target, "score": score}).dropna()
    labels = values["target"].astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC 계산에는 양성과 음성 표본이 모두 필요합니다.")
    ranks = values["score"].rank(method="average")
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def decile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate outcomes across score deciles for descriptive plotting."""
    data = frame[["피로도지수_점수", *OUTCOME_COLUMNS]].copy()
    data["score_decile"] = pd.cut(
        data["피로도지수_점수"], bins=np.linspace(0, 100, 11),
        labels=range(1, 11), include_lowest=True,
    ).astype(int)
    return (
        data.groupby("score_decile", observed=True)
        .agg(n=("WHIP", "size"), WHIP=("WHIP", "median"), ERA=("ERA", "median"),
             FIP=("FIP", "median"), GS=("GS", "median"))
        .reset_index()
    )


def build_audit_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Build the compact set of metrics published in the report and README."""
    checks = validate_dataset(frame)
    correlations = correlation_summary(frame)
    injury_proxy = frame["부상위험도"].notna() & frame["부상위험도"].gt(0)
    return {
        "dataset": checks,
        "correlations": {
            row["metric"]: {
                "overall_r": round(float(row["overall_r"]), 4),
                "within_player_r": round(float(row["within_player_r"]), 4),
            }
            for row in correlations.to_dict(orient="records")
        },
        "classification_diagnostics": {
            "recovery_failure_auc": round(binary_auc(frame["회복실패여부"], frame["피로도지수"]), 4),
            "injury_proxy_auc": round(binary_auc(injury_proxy, frame["피로도지수"]), 4),
            "injury_proxy_definition": "부상위험도가 결측이 아니고 0보다 큰 행",
        },
        "interpretation": {
            "score": "피로도지수_점수는 피로도지수의 전체 표본 내 백분위 순위",
            "scope": "기술적 모니터링 신호이며 생리적 피로·부상·교체 시점을 진단하지 않음",
        },
    }
