"""Build appearance-level process metrics from the licensed KBO PBP dataset.

The upstream dataset is pinned by Git revision and SHA-256. Raw parquet files
are cached outside version control; only the compact, matched appearance table
is published in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue import load_dataset
from kbo_fatigue.appearances import validate_appearances


REVISION = "6afc8af044e3bba5f326b688e8cb41d7ff7065ec"
BASE_URL = "https://huggingface.co/datasets/slothman3878/kbo_playbyplay/resolve"
FILES = {
    2023: (
        "v0/kbo_pbp_2023.parquet",
        "818f6016655b02fe48b8118281d1b04bfe3548d376fdc70131a41ea539341edb",
    ),
    2024: (
        "v0/kbo_pbp_2024.parquet",
        "8332cd716cf0126a4ab0bf390383f43deff22ab320a57fb70d02b31025bdf553",
    ),
}
RAW_COLUMNS = [
    "game_pk", "game_date", "home_team", "away_team", "inning", "inning_topbot",
    "at_bat_number", "pitch_number", "pitcher", "pitcher_name", "home_score",
    "away_score", "pitch_result", "type", "pitch_type", "release_speed_kmh",
    "plate_x", "plate_z", "sz_top", "sz_bot", "outs_when_up", "on_1b",
    "on_2b", "on_3b", "post_home_score", "post_away_score", "post_outs",
    "runs_scored", "post_on_1b", "post_on_2b", "post_on_3b", "release_pos_x",
    "release_pos_z",
]
HARD_PITCHES = {"FF", "SI", "FC"}
TEAM_NAMES = {
    "HH": "한화", "HT": "KIA", "KT": "KT", "LG": "LG", "LT": "롯데",
    "NC": "NC", "OB": "두산", "SK": "SSG", "SS": "삼성", "WO": "키움",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_source_files(cache_dir: Path, download: bool) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for year, (remote_path, expected_hash) in FILES.items():
        path = cache_dir / Path(remote_path).name
        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f"{path}가 없습니다. --download를 사용하거나 원본 parquet를 배치하세요."
                )
            url = f"{BASE_URL}/{REVISION}/{remote_path}?download=true"
            temporary = path.with_suffix(".download")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{year} 원본 SHA-256 불일치: expected={expected_hash}, actual={actual_hash}"
            )
        paths.append(path)
    return paths


def late_velocity_delta(group: pd.DataFrame) -> float:
    hard = group.loc[group["is_hard"] & group["release_speed_kmh"].notna()]
    if len(hard) < 6:
        return float("nan")
    segment = max(3, len(hard) // 3)
    return float(
        hard.tail(segment)["release_speed_kmh"].mean()
        - hard.head(segment)["release_speed_kmh"].mean()
    )


def base_state(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    """Encode occupied bases as the conventional 0–7 RE24 state."""
    return (
        frame[f"{prefix}on_1b"].notna().astype(int)
        + 2 * frame[f"{prefix}on_2b"].notna().astype(int)
        + 4 * frame[f"{prefix}on_3b"].notna().astype(int)
    )


def build_plate_appearances(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build terminal-event plate appearances and a descriptive RE24 table.

    A plate appearance is credited to its terminal-event pitcher, including
    no-pitch walks when supplied by the new-season path.
    The run expectancy table is descriptive of the combined public-data run
    environment; it is not a win-probability or physiological-fatigue model.
    """
    keys = ["game_pk", "at_bat_number"]
    first = pitches.drop_duplicates(keys, keep="first")
    last = pitches.drop_duplicates(keys, keep="last")
    appearances = first[
        keys + [
            "game_date", "home_team", "away_team", "inning", "inning_topbot",
            "outs_when_up", "on_1b", "on_2b", "on_3b", "home_score",
            "away_score",
        ]
    ].merge(
        last[
            keys + [
                "pitcher", "pitcher_name", "post_home_score", "post_away_score",
                "post_outs", "runs_scored", "post_on_1b", "post_on_2b",
                "post_on_3b",
            ]
        ],
        on=keys,
        validate="one_to_one",
    )
    appearances["base_state"] = base_state(appearances)
    appearances["post_base_state"] = base_state(appearances, "post_")
    appearances["batting_score_before"] = np.where(
        appearances["inning_topbot"].eq("top"),
        appearances["away_score"], appearances["home_score"],
    )
    appearances["batting_score_after"] = np.where(
        appearances["inning_topbot"].eq("top"),
        appearances["post_away_score"], appearances["post_home_score"],
    )
    half_inning = ["game_pk", "inning", "inning_topbot"]
    appearances["half_inning_final_score"] = appearances.groupby(
        half_inning, sort=False
    )["batting_score_after"].transform("max")
    appearances["runs_to_inning_end"] = (
        appearances["half_inning_final_score"]
        - appearances["batting_score_before"]
    )
    expectancy_table = (
        appearances.groupby(["outs_when_up", "base_state"], sort=True)
        .agg(
            run_expectancy=("runs_to_inning_end", "mean"),
            plate_appearances=("runs_to_inning_end", "size"),
        )
        .reset_index()
    )
    expectancy_table["base_state_label"] = expectancy_table["base_state"].map(
        lambda state: "".join(
            base if int(state) & bit else "-"
            for bit, base in ((1, "1"), (2, "2"), (4, "3"))
        )
    )
    expectancy = expectancy_table.set_index(
        ["outs_when_up", "base_state"]
    )["run_expectancy"]
    expected_states = pd.MultiIndex.from_product(
        [range(3), range(8)], names=["outs_when_up", "base_state"]
    )
    if not expectancy.index.equals(expected_states):
        missing = expected_states.difference(expectancy.index).tolist()
        raise ValueError(f"RE24 base-out states are incomplete: {missing}")

    appearances["run_expectancy_before"] = pd.MultiIndex.from_frame(
        appearances[["outs_when_up", "base_state"]]
    ).map(expectancy)
    post_index = pd.MultiIndex.from_frame(
        appearances[["post_outs", "post_base_state"]].rename(
            columns={"post_outs": "outs_when_up", "post_base_state": "base_state"}
        )
    )
    appearances["run_expectancy_after"] = post_index.map(expectancy)
    appearances.loc[
        appearances["post_outs"].ge(3), "run_expectancy_after"
    ] = 0.0
    if appearances[["run_expectancy_before", "run_expectancy_after"]].isna().any().any():
        raise ValueError("RE24 lookup produced a missing run expectancy.")
    appearances["re24_allowed"] = (
        appearances["runs_scored"]
        + appearances["run_expectancy_after"]
        - appearances["run_expectancy_before"]
    )
    appearances["팀"] = np.where(
        appearances["inning_topbot"].eq("top"),
        appearances["home_team"], appearances["away_team"],
    )
    appearances["팀"] = appearances["팀"].map(TEAM_NAMES)
    appearances["날짜"] = pd.to_datetime(appearances["game_date"], errors="raise")
    return appearances, expectancy_table


def aggregate_appearance(group: pd.DataFrame, *, include_no_pitch_events: bool = False) -> pd.Series:
    entry = group.iloc[0]
    source_games = int(group["game_pk"].nunique())
    pitcher_ids = int(group["pitcher"].nunique())
    if include_no_pitch_events:
        group = group.loc[group["pitch_number"].gt(0)]
    tracked = group["release_speed_kmh"].notna()
    hard = group["is_hard"] & tracked
    located = group[["plate_x", "plate_z", "sz_bot", "sz_top"]].notna().all(axis=1)
    first_pitch = group["pitch_number"].eq(1)
    hard_release = group.loc[
        group["is_hard"]
        & group[["release_pos_x", "release_pos_z"]].notna().all(axis=1)
    ]
    if len(hard_release) >= 5:
        release_side = float(hard_release["release_pos_x"].median())
        release_height = float(hard_release["release_pos_z"].median())
        radial_inches = 12 * np.sqrt(
            hard_release["release_pos_x"].sub(release_side).pow(2)
            + hard_release["release_pos_z"].sub(release_height).pow(2)
        )
        release_dispersion = float(np.sqrt(radial_inches.pow(2).mean()))
    else:
        release_side = float("nan")
        release_height = float("nan")
        release_dispersion = float("nan")
    return pd.Series(
        {
            "pbp_source_games": source_games,
            "pbp_pitcher_ids": pitcher_ids,
            "pbp_pitch_count": int(len(group)),
            "pbp_tracked_pitches": int(tracked.sum()),
            "pbp_avg_velocity": float(group.loc[tracked, "release_speed_kmh"].mean()),
            "pbp_hard_velocity": float(group.loc[hard, "release_speed_kmh"].mean()),
            "pbp_hard_usage": float(group["is_hard"].mean()),
            "pbp_csw_rate": float(group["is_csw"].mean()),
            "pbp_zone_rate": float(group.loc[located, "is_zone"].mean()),
            "pbp_first_pitch_strike_rate": float(
                group.loc[first_pitch, "is_first_pitch_strike"].mean()
            ),
            "pbp_late_velocity_delta": late_velocity_delta(group),
            "pbp_high_pressure_share": float(group["is_high_pressure"].mean()),
            "pbp_entry_inning": int(entry["inning"]),
            "pbp_entry_outs": int(entry["outs_when_up"]),
            "pbp_entry_runners": int(int(entry["base_state"]).bit_count()),
            "pbp_entry_run_margin": int(entry["defense_run_margin"]),
            "pbp_entry_base_out_re": float(entry["run_expectancy_before"]),
            "pbp_close_late_entry": bool(entry["is_close_late"]),
            "pbp_hard_release_count": int(len(hard_release)),
            "pbp_hard_release_side_ft": release_side,
            "pbp_hard_release_height_ft": release_height,
            "pbp_hard_release_dispersion_in": release_dispersion,
        }
    )


def aggregate_game_appearances(
    pitches: pd.DataFrame, plate_appearances: pd.DataFrame
) -> pd.DataFrame:
    """New-season contract: one source pitcher per source game, never per date."""
    keys = ["game_pk", "pitcher"]
    metadata = ["pitcher_name", "game_date", "팀"]
    if pitches[keys + metadata].isna().any().any():
        raise ValueError("Missing PBP game/player identity or metadata")
    if plate_appearances[keys + ["re24_allowed"]].isna().any().any():
        raise ValueError("Missing PBP plate-appearance identity or RE24")
    pitcher_keys = pd.MultiIndex.from_frame(pitches[keys].drop_duplicates())
    pa_keys = pd.MultiIndex.from_frame(plate_appearances[keys].drop_duplicates())
    if not pa_keys.isin(pitcher_keys).all():
        raise ValueError("Plate appearances reference an unknown game/pitcher")
    grouped = pitches.groupby(keys, sort=True)
    if grouped[metadata].nunique().gt(1).any().any():
        raise ValueError("Conflicting PBP metadata within one game appearance")
    metrics = grouped.apply(
        lambda group: aggregate_appearance(group.assign(
            game_pk=group.name[0], pitcher=group.name[1],
        ), include_no_pitch_events=True), include_groups=False,
    ).reset_index()
    result = grouped[metadata].first().reset_index().merge(
        metrics, on=keys, validate="one_to_one",
    )
    re24 = plate_appearances.groupby(keys).agg(
        pbp_batters_faced=("re24_allowed", "size"),
        pbp_re24_allowed=("re24_allowed", "sum"),
    ).reset_index()
    result = result.merge(re24, on=keys, how="left", validate="one_to_one")
    result[["pbp_batters_faced", "pbp_re24_allowed"]] = result[
        ["pbp_batters_faced", "pbp_re24_allowed"]
    ].fillna(0)
    result["pbp_re24_allowed_per_bf"] = result["pbp_re24_allowed"].div(
        result["pbp_batters_faced"].replace(0, np.nan)
    )
    result["GameID"] = "pbp:game:" + result["game_pk"].astype(str)
    result["PlayerID"] = "pbp:player:" + result["pitcher"].astype(str)
    result = result.rename(columns={"pitcher_name": "Name", "game_date": "Date", "팀": "Team"})
    return validate_appearances(result)


def build_features(
    source_paths: list[Path], canonical_path: Path | None, *, game_level: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pitches = pd.concat(
        [pd.read_parquet(path, columns=RAW_COLUMNS) for path in source_paths],
        ignore_index=True,
    )
    # Keep the historical reproduction unchanged. New seasons must retain
    # no-pitch terminal events for batters faced and run expectancy.
    if not game_level:
        pitches = pitches.loc[pitches["pitch_number"].gt(0)].copy()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"], errors="raise")
    if game_level:
        pitch_key = ["game_pk", "at_bat_number", "pitch_number"]
        if pitches[pitch_key + ["pitcher", "pitcher_name", "game_date"]].isna().any().any():
            raise ValueError("Missing PBP pitch identity")
        if pitches.duplicated(pitch_key).any():
            raise ValueError("Duplicate PBP pitch key; overlapping source files must be resolved")
        if pitches["pitch_number"].lt(0).any():
            raise ValueError("Negative PBP pitch number")
        zero_keys = pitches.loc[pitches.pitch_number.eq(0), ["game_pk", "at_bat_number"]]
        thrown_keys = pitches.loc[pitches.pitch_number.gt(0), ["game_pk", "at_bat_number"]]
        if not zero_keys.merge(thrown_keys, on=["game_pk", "at_bat_number"]).empty:
            raise ValueError("Ambiguous zero-pitch event within a pitched plate appearance")
    if not game_level and not pitches["game_date"].dt.year.isin([2023, 2024]).all():
        raise ValueError("Legacy date aggregation is frozen to 2023–2024; use game_level=True")
    pitches = pitches.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"], kind="stable"
    )
    plate_appearances, expectancy_table = build_plate_appearances(pitches)
    expectancy = expectancy_table.set_index(
        ["outs_when_up", "base_state"]
    )["run_expectancy"]
    pitches["base_state"] = base_state(pitches)
    pitches["run_expectancy_before"] = pd.MultiIndex.from_frame(
        pitches[["outs_when_up", "base_state"]]
    ).map(expectancy)
    pitches["is_hard"] = pitches["pitch_type"].isin(HARD_PITCHES)
    pitches["is_csw"] = pitches["pitch_result"].isin(["T", "S", "V"])
    pitches["is_zone"] = (
        pitches["plate_x"].between(-0.83, 0.83)
        & pitches["plate_z"].between(pitches["sz_bot"], pitches["sz_top"])
    )
    pitches["is_first_pitch_strike"] = pitches["type"].isin(["S", "X"])
    pitches["is_high_pressure"] = (
        pitches["inning"].ge(7)
        & pitches["home_score"].sub(pitches["away_score"]).abs().le(2)
    )
    pitches["defense_run_margin"] = np.where(
        pitches["inning_topbot"].eq("top"),
        pitches["home_score"].sub(pitches["away_score"]),
        pitches["away_score"].sub(pitches["home_score"]),
    )
    pitches["is_close_late"] = (
        pitches["inning"].ge(7) & pitches["defense_run_margin"].abs().le(2)
    )
    pitches["팀"] = np.where(
        pitches["inning_topbot"].eq("top"), pitches["home_team"], pitches["away_team"]
    )
    pitches["팀"] = pitches["팀"].map(TEAM_NAMES)
    if pitches["팀"].isna().any():
        raise ValueError("매핑되지 않은 PBP 팀 코드가 있습니다.")

    if game_level:
        return aggregate_game_appearances(pitches, plate_appearances), expectancy_table

    # Historical reproduction only. Its 36 multi-game rows are intentionally
    # unchanged; do not extend this date-level contract to future seasons.
    appearance = (
        pitches.groupby(["pitcher_name", "game_date", "팀"], sort=False, dropna=False)
        .apply(aggregate_appearance, include_groups=False)
        .reset_index()
        .rename(columns={"pitcher_name": "선수", "game_date": "날짜"})
    )
    re24 = (
        plate_appearances.groupby(["pitcher_name", "날짜", "팀"], sort=False)
        .agg(
            pbp_batters_faced=("re24_allowed", "size"),
            pbp_re24_allowed=("re24_allowed", "sum"),
        )
        .reset_index()
        .rename(columns={"pitcher_name": "선수"})
    )
    appearance = appearance.merge(
        re24, on=["선수", "날짜", "팀"], how="left", validate="one_to_one"
    )
    appearance["pbp_batters_faced"] = appearance["pbp_batters_faced"].fillna(0)
    appearance["pbp_re24_allowed"] = appearance["pbp_re24_allowed"].fillna(0.0)
    appearance["pbp_re24_allowed_per_bf"] = appearance["pbp_re24_allowed"].div(
        appearance["pbp_batters_faced"].replace(0, np.nan)
    )
    canonical = load_dataset(canonical_path)
    canonical_keys = canonical.loc[
        canonical["연도"].isin(FILES),
        ["선수", "날짜", "연도", "팀", "보직", "투구수"],
    ]
    matched = canonical_keys.merge(
        appearance, on=["선수", "날짜", "팀"], how="inner", validate="one_to_one"
    )
    matched["pbp_pitch_count_difference"] = (
        matched["pbp_pitch_count"] - matched["투구수"]
    )
    matched = matched.drop(columns="투구수").sort_values(["선수", "날짜"])
    integer_columns = [
        "pbp_source_games", "pbp_pitcher_ids", "pbp_pitch_count", "pbp_tracked_pitches",
        "pbp_pitch_count_difference", "pbp_entry_inning", "pbp_entry_outs",
        "pbp_entry_runners", "pbp_entry_run_margin", "pbp_batters_faced",
        "pbp_hard_release_count",
    ]
    for column in integer_columns:
        matched[column] = matched[column].astype(int)
    numeric = matched.select_dtypes(include=["float"]).columns
    matched.loc[:, numeric] = matched.loc[:, numeric].round(6)
    expectancy_table["run_expectancy"] = expectancy_table["run_expectancy"].round(6)
    expectancy_table = expectancy_table[
        [
            "outs_when_up", "base_state", "base_state_label", "run_expectancy",
            "plate_appearances",
        ]
    ]
    return matched.reset_index(drop=True), expectancy_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache" / "kbo_playbyplay"
    )
    parser.add_argument(
        "--canonical", type=Path,
        default=ROOT / "data" / "final" / "fatigue_with_index.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "external" / "pbp_appearance_2023_2024.csv",
    )
    parser.add_argument(
        "--re24-output", type=Path,
        default=ROOT / "data" / "external" / "re24_matrix_2023_2024.csv",
    )
    args = parser.parse_args()
    sources = ensure_source_files(args.cache_dir, args.download)
    output, expectancy_table = build_features(sources, args.canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.re24_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8")
    expectancy_table.to_csv(args.re24_output, index=False, encoding="utf-8")
    try:
        destination = args.output.relative_to(ROOT)
    except ValueError:
        destination = args.output
    print(f"Built {len(output):,} matched appearances: {destination}")
    print(f"Built {len(expectancy_table)} RE24 states: {args.re24_output}")


if __name__ == "__main__":
    main()
