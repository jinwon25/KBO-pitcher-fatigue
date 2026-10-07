"""Validate and join a new season in isolation. Never promotes a season to final."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue.appearances import merge_appearances, validate_appearances


def stage_season(inputs: list[Path], features: list[Path], year: int, output: Path) -> dict:
    if year <= 2024:
        raise ValueError("2020–2024 is frozen; stage only seasons after 2024")
    allowed = (ROOT / "data" / "staging" / str(year)).resolve()
    output = output.resolve()
    if not output.is_relative_to(allowed) or output.suffix != ".csv":
        raise ValueError(f"Output must be a CSV under {allowed}")
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError("Use a new snapshot filename; existing snapshots are immutable")
    if not inputs:
        raise ValueError("At least one input is required")
    read = lambda path: pd.read_csv(path, dtype={"PlayerID": "string", "GameID": "string"})
    frame = validate_appearances(pd.concat([read(p) for p in inputs], ignore_index=True))
    for path in features:
        frame = merge_appearances(frame, read(path))
    if not pd.to_datetime(frame["Date"]).dt.year.eq(year).all():
        raise ValueError("Input contains another season")
    frame = frame.sort_values(["Date", "GameID", "PlayerID"]).reset_index(drop=True)
    audit = {
        "year": year, "rows": len(frame), "games": int(frame.GameID.nunique()),
        "players": int(frame.PlayerID.nunique()),
        "date_min": frame.Date.min(), "date_max": frame.Date.max(),
        "identity_validated": True, "season_complete": False, "holdout_ready": False,
        "status": "staging_only_requires_schedule_coverage_and_label_review",
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in inputs + features
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    audit["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--features", type=Path, nargs="*", default=[])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_season(args.input, args.features, args.year, args.output), indent=2))


if __name__ == "__main__":
    main()
