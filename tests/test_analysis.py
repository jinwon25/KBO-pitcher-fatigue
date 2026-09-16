from pathlib import Path

import pandas as pd
import pytest

from kbo_fatigue import binary_auc, build_audit_metrics, decile_summary, load_dataset, validate_dataset


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
