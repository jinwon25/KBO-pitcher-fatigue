"""Pitch-level process metrics used as a licensed external-data extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PBP_METRICS = (
    "pbp_hard_velocity",
    "pbp_hard_usage",
    "pbp_csw_rate",
    "pbp_zone_rate",
    "pbp_first_pitch_strike_rate",
    "pbp_late_velocity_delta",
)
PROCESS_TARGETS = (
    "pbp_csw_rate",
    "pbp_zone_rate",
    "pbp_hard_velocity",
)
PBP_REQUIRED_COLUMNS = {
    "선수", "날짜", "연도", "팀", "보직", "pbp_source_games", "pbp_pitcher_ids",
    "pbp_pitch_count",
    "pbp_tracked_pitches", "pbp_avg_velocity", "pbp_hard_velocity",
    "pbp_hard_usage", "pbp_csw_rate", "pbp_zone_rate",
    "pbp_first_pitch_strike_rate", "pbp_late_velocity_delta",
    "pbp_high_pressure_share", "pbp_pitch_count_difference",
}


def load_pbp_features(path: str | Path) -> pd.DataFrame:
    """Load the tracked appearance-level PBP feature table."""
    frame = pd.read_csv(Path(path))
    missing = PBP_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"PBP 필수 컬럼이 없습니다: {sorted(missing)}")
    frame = frame.copy()
    frame["날짜"] = pd.to_datetime(frame["날짜"], errors="raise")
    frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)
    if frame.duplicated(["선수", "날짜", "팀"]).any():
        raise ValueError("PBP 파생 데이터에 (선수, 날짜, 팀) 중복이 있습니다.")
    return frame.sort_values(["선수", "날짜"], kind="stable").reset_index(drop=True)


def validate_pbp_features(
    pbp: pd.DataFrame, canonical: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Validate source coverage and agreement with the canonical pitch count."""
    missing = PBP_REQUIRED_COLUMNS.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP 필수 컬럼이 없습니다: {sorted(missing)}")
    checks: dict[str, Any] = {
        "rows": int(len(pbp)),
        "players": int(pbp["선수"].nunique()),
        "date_min": pbp["날짜"].min().date().isoformat(),
        "date_max": pbp["날짜"].max().date().isoformat(),
        "years": sorted(int(year) for year in pbp["연도"].unique()),
        "duplicate_player_dates": int(pbp.duplicated(["선수", "날짜", "팀"]).sum()),
        "multiple_pitcher_id_rows": int(pbp["pbp_pitcher_ids"].gt(1).sum()),
        "pitch_count_exact_rate": float(pbp["pbp_pitch_count_difference"].eq(0).mean()),
        "late_velocity_available_rate": float(
            pbp["pbp_late_velocity_delta"].notna().mean()
        ),
    }
    if canonical is not None:
        eligible = canonical.loc[canonical["연도"].isin(checks["years"])]
        matched = eligible.merge(
            pbp[["선수", "날짜", "팀"]],
            on=["선수", "날짜", "팀"], how="left", indicator=True
        )
        checks["canonical_coverage_rate"] = float(matched["_merge"].eq("both").mean())
        canonical_pitch_count = pbp["pbp_pitch_count"] - pbp["pbp_pitch_count_difference"]
        checks["pitch_count_correlation"] = float(
            pbp["pbp_pitch_count"].corr(canonical_pitch_count)
        )
    if checks["duplicate_player_dates"]:
        raise ValueError("PBP 파생 데이터의 키가 유일하지 않습니다.")
    if not pbp["pbp_csw_rate"].dropna().between(0, 1).all():
        raise ValueError("CSW 비율이 0–1 범위를 벗어났습니다.")
    if not pbp["pbp_zone_rate"].dropna().between(0, 1).all():
        raise ValueError("존 비율이 0–1 범위를 벗어났습니다.")
    return checks


def add_pbp_context(canonical: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Merge process metrics and calculate prior-five personal baselines."""
    external_columns = [column for column in pbp.columns if column not in {"연도", "보직"}]
    data = canonical.merge(
        pbp[external_columns],
        on=["선수", "날짜", "팀"], how="left", validate="one_to_one",
    ).sort_values(["선수", "날짜"], kind="stable")
    grouped = data.groupby(["선수", "보직"], sort=False)
    for metric in PBP_METRICS:
        baseline = grouped[metric].transform(
            lambda values: values.shift(1).rolling(5, min_periods=3).mean()
        )
        data[f"{metric}_prior5"] = baseline
        data[f"{metric}_vs_prior5"] = data[metric] - baseline
    return data.reset_index(drop=True)


def _cluster_bootstrap_delta(
    subset: pd.DataFrame,
    target: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    groups = [group for _, group in subset.groupby("선수", sort=False)]
    if not groups or iterations <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample = pd.concat(
            [groups[index] for index in rng.integers(0, len(groups), len(groups))],
            ignore_index=True,
        )
        medians = sample.groupby("above_72_6")[target].median()
        estimates.append(float(medians.get(True, np.nan) - medians.get(False, np.nan)))
    low, high = np.nanquantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def process_signal_summary(
    canonical: pd.DataFrame,
    pbp: pd.DataFrame,
    bootstrap_iterations: int = 0,
) -> pd.DataFrame:
    """Compare next same-role pitch process above and below the 72.6 reference."""
    data = add_pbp_context(canonical, pbp)
    grouped = data.groupby(["선수", "보직"], sort=False)
    data["next_날짜"] = grouped["날짜"].shift(-1)
    data["next_연도"] = grouped["연도"].shift(-1)
    for metric in PROCESS_TARGETS:
        data[f"next_{metric}"] = grouped[metric].shift(-1)
    data["days_to_next"] = (data["next_날짜"] - data["날짜"]).dt.days
    data["above_72_6"] = data["피로도지수_점수"].ge(72.6)
    data = data.loc[
        data["next_연도"].eq(data["연도"]) & data["days_to_next"].between(1, 30)
    ]

    rows: list[dict[str, Any]] = []
    for role_index, role in enumerate(("SP", "RP")):
        for metric_index, metric in enumerate(PROCESS_TARGETS):
            target = f"next_{metric}"
            subset = data.loc[
                data["보직"].eq(role) & data[target].notna(),
                ["선수", "피로도지수_점수", "above_72_6", target],
            ]
            medians = subset.groupby("above_72_6")[target].median()
            low_median = float(medians.get(False, np.nan))
            high_median = float(medians.get(True, np.nan))
            ci_low, ci_high = _cluster_bootstrap_delta(
                subset,
                target,
                bootstrap_iterations,
                seed=42 + role_index * 10 + metric_index,
            )
            rows.append(
                {
                    "role": role,
                    "metric": metric,
                    "n": int(len(subset)),
                    "players": int(subset["선수"].nunique()),
                    "score_r": float(subset["피로도지수_점수"].corr(subset[target])),
                    "below_72_6_median": low_median,
                    "above_72_6_median": high_median,
                    "high_minus_low": high_median - low_median,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def within_appearance_velocity_summary(
    canonical: pd.DataFrame,
    pbp: pd.DataFrame,
    bootstrap_iterations: int = 0,
) -> pd.DataFrame:
    """Summarize hard-pitch velocity change from early to late appearance."""
    data = add_pbp_context(canonical, pbp)
    data = data.loc[data["pbp_late_velocity_delta"].notna()].copy()
    data["above_72_6"] = data["피로도지수_점수"].ge(72.6)
    rows: list[dict[str, Any]] = []
    for index, role in enumerate(("SP", "RP")):
        subset = data.loc[data["보직"].eq(role)]
        medians = subset.groupby("above_72_6")["pbp_late_velocity_delta"].median()
        low_median = float(medians.get(False, np.nan))
        high_median = float(medians.get(True, np.nan))
        ci_low, ci_high = _cluster_bootstrap_delta(
            subset,
            "pbp_late_velocity_delta",
            bootstrap_iterations,
            seed=84 + index,
        )
        rows.append(
            {
                "role": role,
                "n": int(len(subset)),
                "players": int(subset["선수"].nunique()),
                "score_r": float(
                    subset["피로도지수_점수"].corr(subset["pbp_late_velocity_delta"])
                ),
                "below_72_6_median": low_median,
                "above_72_6_median": high_median,
                "high_minus_low": high_median - low_median,
                "cluster_bootstrap_ci_low": ci_low,
                "cluster_bootstrap_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)
