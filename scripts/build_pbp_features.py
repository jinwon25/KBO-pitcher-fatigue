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
    "plate_x", "plate_z", "sz_top", "sz_bot",
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


def aggregate_appearance(group: pd.DataFrame) -> pd.Series:
    tracked = group["release_speed_kmh"].notna()
    hard = group["is_hard"] & tracked
    located = group[["plate_x", "plate_z", "sz_bot", "sz_top"]].notna().all(axis=1)
    first_pitch = group["pitch_number"].eq(1)
    return pd.Series(
        {
            "pbp_source_games": int(group["game_pk"].nunique()),
            "pbp_pitcher_ids": int(group["pitcher"].nunique()),
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
        }
    )


def build_features(source_paths: list[Path], canonical_path: Path) -> pd.DataFrame:
    pitches = pd.concat(
        [pd.read_parquet(path, columns=RAW_COLUMNS) for path in source_paths],
        ignore_index=True,
    )
    pitches = pitches.loc[pitches["pitch_number"].gt(0)].copy()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"], errors="raise")
    pitches = pitches.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"], kind="stable"
    )
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
    pitches["팀"] = np.where(
        pitches["inning_topbot"].eq("top"), pitches["home_team"], pitches["away_team"]
    )
    pitches["팀"] = pitches["팀"].map(TEAM_NAMES)
    if pitches["팀"].isna().any():
        raise ValueError("매핑되지 않은 PBP 팀 코드가 있습니다.")

    appearance = (
        pitches.groupby(["pitcher_name", "game_date", "팀"], sort=False, dropna=False)
        .apply(aggregate_appearance, include_groups=False)
        .reset_index()
        .rename(columns={"pitcher_name": "선수", "game_date": "날짜"})
    )
    canonical = load_dataset(canonical_path)
    canonical_keys = canonical.loc[
        canonical["연도"].isin(FILES),
        ["선수", "날짜", "연도", "팀", "보직", "투구수"],
    ]
    matched = canonical_keys.merge(appearance, on=["선수", "날짜", "팀"], how="inner")
    matched["pbp_pitch_count_difference"] = (
        matched["pbp_pitch_count"] - matched["투구수"]
    )
    matched = matched.drop(columns="투구수").sort_values(["선수", "날짜"])
    integer_columns = [
        "pbp_source_games", "pbp_pitcher_ids", "pbp_pitch_count", "pbp_tracked_pitches",
        "pbp_pitch_count_difference",
    ]
    for column in integer_columns:
        matched[column] = matched[column].astype(int)
    numeric = matched.select_dtypes(include=["float"]).columns
    matched.loc[:, numeric] = matched.loc[:, numeric].round(6)
    return matched.reset_index(drop=True)


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
    args = parser.parse_args()
    sources = ensure_source_files(args.cache_dir, args.download)
    output = build_features(sources, args.canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Built {len(output):,} matched appearances: {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
