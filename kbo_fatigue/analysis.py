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
FORWARD_FEATURES = (
    "투구수", "이닝", "NP/IP", "투구수_roll3", "이닝_roll3", "휴식일수",
    "연투여부", "연투일수", "연투횟수", "누적연투일수", "휴식대연투비율",
    "GS_roll3", "GS_roll5",
)


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


def add_forward_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach each pitcher's next outcome in the same role.

    Grouping by both player and role keeps starter and relief workloads on
    separate timelines, including seasons in which a pitcher changes roles.
    """
    data = frame.sort_values(["선수", "날짜"], kind="stable").copy()
    grouped = data.groupby(["선수", "보직"], sort=False)
    for column in ("날짜", *OUTCOME_COLUMNS):
        data[f"next_{column}"] = grouped[column].shift(-1)
    data["days_to_next"] = (data["next_날짜"] - data["날짜"]).dt.days
    data["same_role_next"] = data["next_날짜"].notna()
    data["above_72_6"] = data["피로도지수_점수"].ge(72.6)
    for window in (3, 5):
        data[f"GS_roll{window}"] = grouped["GS"].transform(
            lambda values: values.rolling(window, min_periods=2).mean()
        )
    return data


def _within_group_correlation(
    frame: pd.DataFrame, x: str, y: str, groups: list[str]
) -> float:
    centered_x = frame[x] - frame.groupby(groups)[x].transform("mean")
    centered_y = frame[y] - frame.groupby(groups)[y].transform("mean")
    return float(centered_x.corr(centered_y))


def forward_validation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize same-game and next-appearance associations by pitcher role."""
    data = add_forward_features(frame)
    valid = data.loc[data["same_role_next"] & data["next_GS"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for role in ("ALL", "SP", "RP"):
        subset = valid if role == "ALL" else valid.loc[valid["보직"] == role]
        low = subset.loc[~subset["above_72_6"], "next_GS"]
        high = subset.loc[subset["above_72_6"], "next_GS"]
        rows.append(
            {
                "role": role,
                "n": int(len(subset)),
                "players": int(subset["선수"].nunique()),
                "same_game_gs_r": float(subset["피로도지수_점수"].corr(subset["GS"])),
                "next_appearance_gs_r": float(
                    subset["피로도지수_점수"].corr(subset["next_GS"])
                ),
                "within_player_season_next_gs_r": _within_group_correlation(
                    subset, "피로도지수_점수", "next_GS", ["선수", "연도"]
                ),
                "below_72_6_next_gs_median": float(low.median()),
                "above_72_6_next_gs_median": float(high.median()),
                "above_72_6_n": int(high.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def forward_decile_summary(
    frame: pd.DataFrame,
    roles: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate next same-role appearance outcomes by current-score decile."""
    data = add_forward_features(frame)
    data = data.loc[data["same_role_next"] & data["next_GS"].notna()].copy()
    if roles is not None:
        data = data.loc[data["보직"].isin(roles)]
    if years is not None:
        data = data.loc[data["연도"].isin(years)]
    data["score_decile"] = pd.cut(
        data["피로도지수_점수"], bins=np.linspace(0, 100, 11),
        labels=range(1, 11), include_lowest=True,
    ).astype(int)
    return (
        data.groupby("score_decile", observed=True)
        .agg(
            n=("next_GS", "size"),
            WHIP=("next_WHIP", "median"), ERA=("next_ERA", "median"),
            FIP=("next_FIP", "median"), GS=("next_GS", "median"),
        )
        .reset_index()
    )


def _ridge_predict(
    train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...], alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train.loc[:, features].to_numpy(dtype=float)
    x_test = test.loc[:, features].to_numpy(dtype=float)
    y_train = train["next_GS"].to_numpy(dtype=float)
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales == 0] = 1
    design_train = np.column_stack([np.ones(len(train)), (x_train - means) / scales])
    design_test = np.column_stack([np.ones(len(test)), (x_test - means) / scales])
    penalty = np.eye(design_train.shape[1]) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(
        design_train.T @ design_train + penalty,
        design_train.T @ y_train,
    )
    return design_test @ coefficients, coefficients


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    denominator = np.square(actual - actual.mean()).sum()
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1 - np.square(residual).sum() / denominator),
    }


def temporal_starter_validation(frame: pd.DataFrame) -> dict[str, Any]:
    """Run a time-ordered next-start baseline using workload and recent form.

    Alpha is selected on 2023 after training through 2022. The final model is
    refit through 2023 and evaluated once on 2024.
    """
    data = add_forward_features(frame)
    required = [*FORWARD_FEATURES, "next_GS", "next_날짜"]
    starters = data.loc[
        data["same_role_next"] & data["보직"].eq("SP") & data["next_GS"].notna()
    ].dropna(subset=required)

    train_early = starters.loc[starters["next_날짜"].dt.year <= 2022]
    validation = starters.loc[starters["next_날짜"].dt.year == 2023]
    candidates = (0.1, 1.0, 10.0, 100.0, 1000.0)
    validation_mae: dict[float, float] = {}
    for alpha in candidates:
        prediction, _ = _ridge_predict(train_early, validation, FORWARD_FEATURES, alpha)
        validation_mae[alpha] = float(
            np.abs(validation["next_GS"].to_numpy(dtype=float) - prediction).mean()
        )
    selected_alpha = min(validation_mae, key=validation_mae.get)

    train = starters.loc[starters["next_날짜"].dt.year <= 2023]
    test = starters.loc[starters["next_날짜"].dt.year == 2024]
    prediction, coefficients = _ridge_predict(train, test, FORWARD_FEATURES, selected_alpha)
    actual = test["next_GS"].to_numpy(dtype=float)
    baseline = np.repeat(train["next_GS"].mean(), len(test))
    model_metrics = _regression_metrics(actual, prediction)
    baseline_metrics = _regression_metrics(actual, baseline)

    return {
        "target": "next same-role starter GS",
        "features": list(FORWARD_FEATURES),
        "validation_year": 2023,
        "test_year": 2024,
        "selected_alpha": float(selected_alpha),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "model": {key: round(value, 4) for key, value in model_metrics.items()},
        "mean_baseline": {key: round(value, 4) for key, value in baseline_metrics.items()},
        "mae_improvement_pct": round(
            100 * (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"],
            2,
        ),
        "standardized_coefficients": {
            feature: round(float(value), 4)
            for feature, value in zip(FORWARD_FEATURES, coefficients[1:])
        },
        "interpretation": (
            "탐색적 시간 외 검증이며 운영 예측 모델이 아님. GS는 선발 평가에만 사용."
        ),
    }


def build_audit_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Build the compact set of metrics published in the report and README."""
    checks = validate_dataset(frame)
    correlations = correlation_summary(frame)
    injury_proxy = frame["부상위험도"].notna() & frame["부상위험도"].gt(0)
    forward = forward_validation_summary(frame)
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
        "forward_validation": {
            row["role"]: {
                key: (round(float(value), 4) if isinstance(value, float) else value)
                for key, value in row.items()
                if key != "role"
            }
            for row in forward.to_dict(orient="records")
        },
        "temporal_starter_model": temporal_starter_validation(frame),
        "interpretation": {
            "score": "피로도지수_점수는 피로도지수의 전체 표본 내 백분위 순위",
            "scope": "기술적 모니터링 신호이며 생리적 피로·부상·교체 시점을 진단하지 않음",
        },
    }
