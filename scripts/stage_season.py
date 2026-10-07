"""Validate and join a new season in isolation. Never promotes a season to final."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue.appearances import merge_appearances, validate_appearances
from kbo_fatigue.snapshots import publish_snapshot, staging_csv_path


def stage_season(inputs: list[Path], features: list[Path], year: int, output: Path) -> dict:
    output = staging_csv_path(ROOT, year, output)
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError("Use a new snapshot filename; existing snapshots are immutable")
    if not inputs:
        raise ValueError("At least one input is required")
    provenance = []

    def read(path: Path) -> pd.DataFrame:
        content = path.read_bytes()
        provenance.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
        # Hash the exact bytes parsed, even if the source changes during the run.
        return pd.read_csv(BytesIO(content), dtype={"PlayerID": "string", "GameID": "string"})
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
        "inputs": provenance,
    }
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    audit["output_sha256"] = hashlib.sha256(content).hexdigest()
    publish_snapshot({
        output: content,
        manifest: (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    })
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
