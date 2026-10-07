"""Reproducible, staging-only public PBP backfill with a schedule coverage audit.

The NAVER schedule endpoint is undocumented, not an official KBO open API.
Network collection is opt-in; immutable local snapshots allow offline rebuilds.
"""
from __future__ import annotations

import argparse
import calendar
import gzip
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue.snapshots import publish_snapshot, staging_csv_path
from scripts.build_pbp_features import BASE_URL, TEAM_NAMES, build_features

SOURCES = {
    2025: ("6afc8af044e3bba5f326b688e8cb41d7ff7065ec",
           "2c824919495809722a5ff0290a823ff9a44d88f61640ad9b288ff3dca2652f2c"),
    2026: ("6afc8af044e3bba5f326b688e8cb41d7ff7065ec",
           "9d330311d28371806028b878191fcc85b9170839c8951b00ff9c64ec8aa28630"),
}
SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"
SCHEDULE_FIELDS = [
    "gameId", "gameDate", "gameDateTime", "homeTeamCode", "awayTeamCode",
    "statusCode", "cancel", "suspended", "roundCode", "timeTbd",
]


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value: dict | list) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": "KBO-pitcher-fatigue research/1.0", "Accept": "application/json",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def source_file(year: int, cache: Path, download: bool) -> Path:
    revision, expected = SOURCES[year]
    path = cache / f"kbo_pbp_{year}.parquet"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path}: use --download to fetch the pinned source")
        content = request_bytes(f"{BASE_URL}/{revision}/v0/{path.name}")
        if digest(content) != expected:
            raise ValueError("Downloaded PBP SHA-256 mismatch")
        publish_snapshot({path: content})
    if digest(path.read_bytes()) != expected:
        raise ValueError("Cached PBP SHA-256 mismatch")
    return path


def schedule_games(snapshot: dict, year: int, through: date) -> pd.DataFrame:
    """Reject missing windows, capped responses, duplicates and malformed metadata."""
    if through.year != year:
        raise ValueError("Schedule cutoff must be in requested season")
    requests = snapshot["requests"]
    expected_windows = {
        (f"{year}-{month:02d}-01", min(through, date(year, month, calendar.monthrange(year, month)[1])).isoformat())
        for month in range(1, through.month + 1)
    }
    windows = set()
    records = []
    for item in requests:
        parsed = urllib.parse.urlsplit(item["url"])
        params = urllib.parse.parse_qs(parsed.query)
        if f"{parsed.scheme}://{parsed.netloc}{parsed.path}" != SCHEDULE_URL:
            raise ValueError("Unexpected schedule endpoint")
        for key, value in {"categoryId": "kbo", "roundCodes": "kbo_r", "page": "1", "size": "500"}.items():
            if params.get(key) != [value]:
                raise ValueError(f"Unexpected schedule parameter: {key}")
        window = (params["fromDate"][0], params["toDate"][0])
        if window in windows:
            raise ValueError("Duplicate schedule window")
        windows.add(window)
        response = item["response"]
        if response.get("success") is not True or response.get("code") != 200:
            raise ValueError("Unsuccessful schedule response")
        result = response["result"]
        games = result["games"]
        # This API can cap both games and gameTotalCount at the requested size.
        if len(games) >= 500 or len(games) != result["gameTotalCount"]:
            raise ValueError("Truncated schedule response; use smaller date windows")
        for game in games:
            if not window[0] <= game["gameDate"] <= window[1]:
                raise ValueError("Schedule game outside requested window")
            records.append({key: game[key] for key in SCHEDULE_FIELDS})
    if windows != expected_windows:
        raise ValueError("Schedule snapshot has incomplete or unexpected date windows")
    frame = pd.DataFrame(records, columns=SCHEDULE_FIELDS)
    if frame.empty or frame.isna().any().any():
        raise ValueError("Empty or incomplete schedule metadata")
    if frame.gameId.duplicated().any() or not frame.roundCode.eq("kbo_r").all():
        raise ValueError("Duplicate game ID or non-regular-season schedule")
    for column in ("cancel", "suspended", "timeTbd"):
        if not frame[column].map(lambda value: isinstance(value, bool)).all():
            raise ValueError(f"Non-boolean schedule {column}")
    for column in ("homeTeamCode", "awayTeamCode"):
        if not frame[column].isin(TEAM_NAMES).all():
            raise ValueError("Unknown schedule team")
    return frame.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


def collect_schedule(year: int, through: date, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("Use a new schedule snapshot filename")
    if through.year != year or through > date.today():
        raise ValueError("Schedule cutoff must be in the season and not in the future")
    requests = []
    for month in range(1, through.month + 1):
        end = min(through, date(year, month, calendar.monthrange(year, month)[1]))
        params = {
            "fields": "basic,schedule,baseball", "upperCategoryId": "kbaseball",
            "categoryId": "kbo", "fromDate": f"{year}-{month:02d}-01",
            "toDate": end.isoformat(), "roundCodes": "kbo_r", "size": 500, "page": 1,
        }
        url = SCHEDULE_URL + "?" + urllib.parse.urlencode(params)
        requests.append({"url": url, "response": json.loads(request_bytes(url))})
    snapshot = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "requests": requests}
    schedule_games(snapshot, year, through)
    publish_snapshot({destination: json_bytes(snapshot)})


def audit_coverage(raw: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    columns = ["game_date", "home_team", "away_team"]
    if raw.groupby("game_pk")[columns].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting PBP game metadata")
    games = raw[["game_pk"] + columns].drop_duplicates().copy()
    games["game_date"] = pd.to_datetime(games.game_date).dt.strftime("%Y-%m-%d")
    matched = games.merge(schedule, left_on="game_pk", right_on="gameId", how="left", validate="one_to_one")
    if matched.gameId.isna().any():
        raise ValueError("PBP game absent from schedule")
    if not (matched.game_date.eq(matched.gameDate) & matched.home_team.eq(matched.homeTeamCode)
            & matched.away_team.eq(matched.awayTeamCode)).all():
        raise ValueError("PBP/schedule date or teams conflict")
    if not (matched.statusCode.eq("RESULT") & ~matched.cancel).all():
        raise ValueError("PBP includes an uncompleted or canceled schedule game")
    completed = schedule.loc[schedule.statusCode.eq("RESULT") & ~schedule.cancel]
    missing = completed.loc[~completed.gameId.isin(games.game_pk)]
    cutoff = games.game_date.max()
    return {
        "pbp_games": len(games), "schedule_completed_games": len(completed),
        "schedule_canceled_games": int(schedule.cancel.sum()),
        "schedule_pending_games": int((~schedule.cancel & ~schedule.statusCode.eq("RESULT")).sum()),
        "missing_completed_games": missing.gameId.tolist(),
        "missing_completed_games_through_pbp_date": missing.loc[missing.gameDate.le(cutoff), "gameId"].tolist(),
        "schedule_completed_coverage": missing.empty,
        "schedule_suspended_game_ids": schedule.loc[schedule.suspended, "gameId"].tolist(),
        "scope": "NAVER regular-season schedule snapshot; not independent official KBO certification",
    }


def backfill(year: int, source: Path, schedule_path: Path, through: date, output: Path) -> dict:
    output = staging_csv_path(ROOT, year, output)
    manifest_path = output.with_suffix(".manifest.json")
    schedule_output = output.with_suffix(".schedule.csv")
    schedule_snapshot = output.with_suffix(".schedule.json.gz")
    re24_output = output.with_suffix(".re24.csv")
    if any(p.exists() for p in (output, manifest_path, schedule_output, schedule_snapshot, re24_output)):
        raise FileExistsError("Use a new snapshot filename")
    # Hash the same in-memory source bytes used by the build, avoiding a reread race.
    source_bytes = source.read_bytes()
    revision, expected = SOURCES[year]
    if digest(source_bytes) != expected:
        raise ValueError("PBP SHA-256 mismatch")
    schedule_bytes = schedule_path.read_bytes()
    if schedule_path.suffix == ".gz":
        schedule_bytes = gzip.decompress(schedule_bytes)
    snapshot = json.loads(schedule_bytes)
    schedule = schedule_games(snapshot, year, through)
    raw = pd.read_parquet(BytesIO(source_bytes))
    if not pd.to_datetime(raw.game_date).dt.year.eq(year).all():
        raise ValueError("PBP contains another season")
    coverage = audit_coverage(raw, schedule)
    features, re24 = build_features([BytesIO(source_bytes)], None, game_level=True)
    if int(features.pbp_pitch_count.sum()) != int(raw.pitch_number.gt(0).sum()):
        raise ValueError("Pitch totals changed during aggregation")
    if int(features.pbp_batters_faced.sum()) != len(raw[["game_pk", "at_bat_number"]].drop_duplicates()):
        raise ValueError("Plate-appearance totals changed during aggregation")
    features = features.merge(schedule[["gameId", "gameDateTime", "timeTbd"]],
                              left_on="game_pk", right_on="gameId", validate="many_to_one").drop(columns="gameId")
    features = features.rename(columns={"gameDateTime": "ScheduledStartKST", "timeTbd": "ScheduledTimeTBD"})
    features = features.sort_values(["Date", "GameID", "PlayerID"]).reset_index(drop=True)
    features["Date"] = pd.to_datetime(features.Date).dt.strftime("%Y-%m-%d")
    features = features.round(6)
    re24 = re24.round(6)
    artifacts = {output: features.to_csv(index=False).encode(),
                 schedule_output: schedule.to_csv(index=False).encode(),
                 schedule_snapshot: gzip.compress(schedule_bytes, mtime=0),
                 re24_output: re24.to_csv(index=False).encode()}
    audit = {
        "year": year, "rows": len(features), "players": int(features.PlayerID.nunique()),
        "date_min": features.Date.min(), "date_max": features.Date.max(),
        "raw_rows": len(raw), "pitches": int(raw.pitch_number.gt(0).sum()),
        "no_pitch_events": int(raw.pitch_number.eq(0).sum()),
        "plate_appearances": int(features.pbp_batters_faced.sum()),
        "same_day_repeat_appearances": int(features.duplicated(["PlayerID", "Date"]).sum()),
        "identity_validated": True, "season_complete": False, "holdout_ready": False,
        "status": "staging_only_requires_official_totals_crosswalk_and_label_review",
        "source": {"dataset": "slothman3878/kbo_playbyplay", "revision": revision,
                   "url": f"{BASE_URL}/{revision}/v0/kbo_pbp_{year}.parquet",
                   "sha256": digest(source_bytes), "license": "CC-BY-4.0",
                   "attribution": "slothman3878; structured derivatives of NAVER Sports/KBO records"},
        "schedule": {"retrieved_at": snapshot["retrieved_at"], "through": through.isoformat(),
                     "snapshot_sha256": digest(schedule_bytes),
                     "urls": [item["url"] for item in snapshot["requests"]]},
        "coverage": coverage,
        "artifacts": {path.name: digest(content) for path, content in artifacts.items()},
        "limitations": ["PBP-derived metrics, not verified official season totals or injury labels",
                        "RE24 is descriptive of this snapshot and must not train a holdout model",
                        "ScheduledStartKST is scheduled local time, not verified actual pitch time",
                        "Cross-source IDs are not mapped; legacy 2025 rows remain unresolved"],
    }
    artifacts[manifest_path] = json_bytes(audit)  # manifest is the completion marker
    publish_snapshot(artifacts)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=sorted(SOURCES), required=True)
    parser.add_argument("--through", type=date.fromisoformat, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/kbo_playbyplay")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--collect-schedule", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Enforce isolation before any network or file writes.
    staging_csv_path(ROOT, args.year, args.output)
    source = source_file(args.year, args.cache_dir, args.download)
    if args.collect_schedule:
        collect_schedule(args.year, args.through, args.schedule)
    print(json.dumps(backfill(args.year, source, args.schedule, args.through, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
