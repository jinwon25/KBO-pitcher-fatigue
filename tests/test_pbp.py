from pathlib import Path

import pandas as pd
import pytest

from kbo_fatigue import (
    add_pbp_context,
    load_dataset,
    load_pbp_features,
    process_signal_summary,
    release_point_summary,
    reliever_re24_decile_summary,
    reliever_re24_summary,
    validate_pbp_features,
    within_appearance_velocity_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "final" / "fatigue_with_index.csv"
PBP = ROOT / "data" / "external" / "pbp_appearance_2023_2024.csv"
RE24_MATRIX = ROOT / "data" / "external" / "re24_matrix_2023_2024.csv"


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
    assert pbp["pbp_batters_faced"].ge(1).all()
    assert pbp["pbp_entry_outs"].between(0, 2).all()
    assert pbp["pbp_entry_runners"].between(0, 3).all()
    assert pbp["pbp_re24_allowed_per_bf"].notna().all()
    assert pbp["pbp_hard_release_count"].ge(0).all()
    assert pbp["pbp_hard_release_dispersion_in"].dropna().ge(0).all()


def test_re24_matrix_contains_all_base_out_states():
    matrix = pd.read_csv(RE24_MATRIX)
    assert len(matrix) == 24
    assert matrix[["outs_when_up", "base_state"]].drop_duplicates().shape[0] == 24
    assert matrix["plate_appearances"].min() == 188
    empty_zero_out = matrix.loc[
        matrix["outs_when_up"].eq(0) & matrix["base_state"].eq(0),
        "run_expectancy",
    ].iloc[0]
    assert empty_zero_out == pytest.approx(0.559513, abs=1e-6)


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


def test_reliever_re24_separates_description_from_forward_validation(datasets):
    canonical, pbp = datasets
    summary = reliever_re24_summary(canonical, pbp).set_index("scope")
    assert summary.loc["all_relief", "n"] == 6_167
    assert summary.loc["close_late_entry", "n"] == 2_659
    assert summary.loc["all_relief", "high_minus_low"] == pytest.approx(
        0.002911, abs=1e-6
    )
    assert summary.loc["all_relief", "score_r"] == pytest.approx(-0.003223, abs=1e-6)

    deciles = reliever_re24_decile_summary(canonical, pbp).set_index(
        ["horizon", "score_decile"]
    )
    same_low = deciles.loc[("same_appearance", 1), "mean_re24_allowed_per_bf"]
    same_high = deciles.loc[("same_appearance", 10), "mean_re24_allowed_per_bf"]
    next_low = deciles.loc[("next_appearance", 1), "mean_re24_allowed_per_bf"]
    next_high = deciles.loc[("next_appearance", 10), "mean_re24_allowed_per_bf"]
    assert same_high - same_low > 0.5
    assert abs(next_high - next_low) < 0.01


def test_release_point_consistency_is_role_and_horizon_specific(datasets):
    canonical, pbp = datasets
    summary = release_point_summary(canonical, pbp).set_index(["role", "horizon"])
    assert summary.loc[("SP", "same_appearance"), "n"] == 1_520
    assert summary.loc[("RP", "same_appearance"), "n"] == 5_199
    assert summary.loc[("RP", "same_appearance"), "high_minus_low_in"] == pytest.approx(
        0.104208, abs=1e-6
    )
    assert summary.loc[("RP", "next_appearance"), "score_r"] == pytest.approx(
        0.013937, abs=1e-6
    )
