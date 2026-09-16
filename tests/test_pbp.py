from pathlib import Path

import pytest

from kbo_fatigue import (
    add_pbp_context,
    load_dataset,
    load_pbp_features,
    process_signal_summary,
    validate_pbp_features,
    within_appearance_velocity_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "final" / "fatigue_with_index.csv"
PBP = ROOT / "data" / "external" / "pbp_appearance_2023_2024.csv"


@pytest.fixture(scope="module")
def datasets():
    return load_dataset(CANONICAL), load_pbp_features(PBP)


def test_external_pbp_contract_and_coverage(datasets):
    canonical, pbp = datasets
    checks = validate_pbp_features(pbp, canonical)
    assert checks["rows"] == 7_924
    assert checks["players"] == 128
    assert checks["years"] == [2023, 2024]
    assert checks["duplicate_player_dates"] == 0
    assert checks["multiple_pitcher_id_rows"] == 0
    assert checks["canonical_coverage_rate"] == pytest.approx(0.9745, abs=1e-4)
    assert checks["pitch_count_correlation"] == pytest.approx(0.9995, abs=1e-4)
    assert checks["pitch_count_exact_rate"] == pytest.approx(0.9937, abs=1e-4)


def test_personal_baseline_excludes_current_appearance(datasets):
    canonical, pbp = datasets
    context = add_pbp_context(canonical, pbp)
    player = context.loc[
        context["pbp_hard_velocity_prior5"].notna() & context["보직"].eq("SP"), "선수"
    ].iloc[0]
    player_rows = context.loc[
        context["선수"].eq(player) & context["보직"].eq("SP")
    ].sort_values("날짜")
    row = player_rows.loc[player_rows["pbp_hard_velocity_prior5"].notna()].iloc[0]
    prior = player_rows.loc[player_rows["날짜"].lt(row["날짜"]), "pbp_hard_velocity"].dropna().tail(5)
    assert row["pbp_hard_velocity_prior5"] == pytest.approx(prior.mean(), abs=1e-6)


def test_pitch_process_follow_up_is_reproducible(datasets):
    canonical, pbp = datasets
    summary = process_signal_summary(canonical, pbp).set_index(["role", "metric"])
    assert summary.loc[("SP", "pbp_csw_rate"), "n"] == 1_398
    assert summary.loc[("RP", "pbp_csw_rate"), "n"] == 6_167
    assert summary.loc[("SP", "pbp_hard_velocity"), "high_minus_low"] == pytest.approx(
        -0.5965, abs=1e-4
    )
    assert summary.loc[("RP", "pbp_zone_rate"), "score_r"] == pytest.approx(
        -0.0248, abs=1e-4
    )


def test_within_appearance_velocity_signal_is_reproducible(datasets):
    canonical, pbp = datasets
    summary = within_appearance_velocity_summary(canonical, pbp).set_index("role")
    assert summary.loc["SP", "n"] == 1_534
    assert summary.loc["SP", "score_r"] == pytest.approx(-0.1265, abs=1e-4)
    assert summary.loc["SP", "below_72_6_median"] == -1.0
    assert summary.loc["SP", "above_72_6_median"] == pytest.approx(-1.2857, abs=1e-4)
