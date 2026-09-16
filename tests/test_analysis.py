from pathlib import Path

import pandas as pd
import pytest

from kbo_fatigue import (
    binary_auc,
    build_audit_metrics,
    decile_summary,
    forward_validation_summary,
    load_dataset,
    temporal_starter_validation,
    validate_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "final" / "fatigue_with_index.csv"


@pytest.fixture(scope="module")
def frame():
    return load_dataset(DATA)


def test_canonical_dataset_contract(frame):
    checks = validate_dataset(frame)
    assert checks["rows"] == 17_528
    assert checks["players"] == 163
    assert checks["date_min"] == "2020-05-05"
    assert checks["date_max"] == "2024-10-01"
    assert checks["years"] == [2020, 2021, 2022, 2023, 2024]
    assert checks["duplicate_player_dates"] == 0
    assert checks["percentile_rank_max_error"] < 1e-9


def test_published_audit_metrics_are_reproducible(frame):
    metrics = build_audit_metrics(frame)
    assert metrics["correlations"]["WHIP"]["overall_r"] == pytest.approx(0.8338, abs=1e-4)
    assert metrics["correlations"]["GS"]["overall_r"] == pytest.approx(-0.3890, abs=1e-4)
    assert metrics["classification_diagnostics"]["recovery_failure_auc"] == pytest.approx(0.5143, abs=1e-4)
    assert metrics["classification_diagnostics"]["injury_proxy_auc"] == pytest.approx(0.5079, abs=1e-4)


def test_decile_summary_is_complete(frame):
    summary = decile_summary(frame)
    assert summary["score_decile"].tolist() == list(range(1, 11))
    assert int(summary["n"].sum()) == len(frame)
    assert not summary.isna().any().any()


def test_binary_auc_handles_ties_and_rejects_one_class():
    assert binary_auc(pd.Series([0, 0, 1, 1]), pd.Series([0, 0, 1, 1])) == 1.0
    assert binary_auc(pd.Series([0, 1]), pd.Series([1, 1])) == 0.5
    with pytest.raises(ValueError):
        binary_auc(pd.Series([1, 1]), pd.Series([0, 1]))


def test_role_specific_forward_validation(frame):
    summary = forward_validation_summary(frame).set_index("role")
    assert summary.loc["SP", "n"] == 3_406
    assert summary.loc["RP", "n"] == 13_870
    assert summary.loc["SP", "next_appearance_gs_r"] == pytest.approx(-0.1243, abs=1e-4)
    assert summary.loc["SP", "below_72_6_next_gs_median"] == 64.0
    assert summary.loc["SP", "above_72_6_next_gs_median"] == 60.0


def test_temporal_starter_model_uses_2024_as_holdout(frame):
    result = temporal_starter_validation(frame)
    assert result["selected_alpha"] == 1000.0
    assert result["train_n"] == 2_452
    assert result["test_n"] == 861
    assert result["model"]["mae"] == pytest.approx(9.7332, abs=1e-4)
    assert result["model"]["r2"] == pytest.approx(0.0841, abs=1e-4)
    assert result["mae_improvement_pct"] == pytest.approx(4.28, abs=0.01)


def test_original_kim_taekyeon_case_is_reproduced(frame):
    row = frame.loc[
        frame["선수"].eq("김택연") & frame["날짜"].eq(pd.Timestamp("2024-08-24"))
    ].iloc[0]
    assert row["피로도지수_점수"] == pytest.approx(99.9772, abs=1e-4)
    assert row["ERA"] == 54.0
    assert row["WHIP"] == 9.0
    assert row["GS"] == 37
