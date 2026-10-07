"""Offline contracts for public backfills and the committed real snapshots."""
import gzip
import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pytest

from kbo_fatigue.appearances import validate_appearances
from scripts import backfill_public_pbp as backfill

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def schedule_snapshot():
    game = dict(gameId="20250101LGKT12025", gameDate="2025-01-01",
                gameDateTime="2025-01-01T14:00:00", homeTeamCode="KT", awayTeamCode="LG",
                statusCode="RESULT", cancel=False, suspended=False, roundCode="kbo_r", timeTbd=False)
    params = dict(categoryId="kbo", roundCodes="kbo_r", page=1, size=500,
                  fromDate="2025-01-01", toDate="2025-01-31")
    return dict(retrieved_at="2026-10-07T00:00:00Z", requests=[dict(
        url=backfill.SCHEDULE_URL + "?" + urlencode(params),
        response=dict(code=200, success=True, result=dict(games=[game], gameTotalCount=1)))])


def test_schedule_requires_every_window_and_no_truncation(schedule_snapshot):
    assert len(backfill.schedule_games(schedule_snapshot, 2025, date(2025, 1, 31))) == 1
    with pytest.raises(ValueError, match="windows"):
        backfill.schedule_games(schedule_snapshot, 2025, date(2025, 2, 28))
    schedule_snapshot["requests"][0]["response"]["result"]["gameTotalCount"] = 2
    with pytest.raises(ValueError, match="Truncated"):
        backfill.schedule_games(schedule_snapshot, 2025, date(2025, 1, 31))


def test_api_cap_is_not_treated_as_complete(schedule_snapshot):
    result = schedule_snapshot["requests"][0]["response"]["result"]
    result["games"] *= 500
    result["gameTotalCount"] = 500
    with pytest.raises(ValueError, match="Truncated"):
        backfill.schedule_games(schedule_snapshot, 2025, date(2025, 1, 31))


def test_duplicate_schedule_ids_and_wrong_teams_fail(schedule_snapshot):
    result = schedule_snapshot["requests"][0]["response"]["result"]
    result["games"] *= 2
    result["gameTotalCount"] = 2
    with pytest.raises(ValueError, match="Duplicate game"):
        backfill.schedule_games(schedule_snapshot, 2025, date(2025, 1, 31))
    result["games"] = [result["games"][0]]
    result["gameTotalCount"] = 1
    result["games"][0]["homeTeamCode"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown schedule team"):
        backfill.schedule_games(schedule_snapshot, 2025, date(2025, 1, 31))


def test_coverage_checks_id_date_teams_and_completed_status(schedule_snapshot):
    schedule = backfill.schedule_games(schedule_snapshot, 2025, date(2025, 1, 31))
    raw = pd.DataFrame([dict(game_pk="20250101LGKT12025", game_date="2025-01-01",
                            home_team="KT", away_team="LG")])
    assert backfill.audit_coverage(raw, schedule)["schedule_completed_coverage"]
    for column, value in [("game_pk", "other"), ("game_date", "2025-01-02"), ("home_team", "SS")]:
        with pytest.raises(ValueError):
            backfill.audit_coverage(raw.assign(**{column: value}), schedule)
    with pytest.raises(ValueError, match="uncompleted or canceled"):
        backfill.audit_coverage(raw, schedule.assign(cancel=True))
    game2 = schedule.iloc[0].copy()
    game2["gameId"] = "20250101LGKT22025"
    audit = backfill.audit_coverage(raw, pd.concat([schedule, pd.DataFrame([game2.to_dict()])]))
    assert audit["missing_completed_games"] == ["20250101LGKT22025"]
    assert not audit["schedule_completed_coverage"]


def test_bad_download_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "request_bytes", lambda url: b"corrupt download")
    with pytest.raises(ValueError, match="SHA-256"):
        backfill.source_file(2025, tmp_path, True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("year,games,rows,no_pitch", [(2025, 720, 7025, 122), (2026, 470, 4594, 51)])
def test_committed_real_backfill_integrity(year, games, rows, no_pitch):
    path = ROOT / f"data/staging/{year}/public_pbp_v1.csv"
    audit = json.loads(path.with_suffix(".manifest.json").read_text())
    for name, expected in audit["artifacts"].items():
        assert hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == expected
    frame = validate_appearances(pd.read_csv(path))
    assert len(frame) == rows == audit["rows"]
    assert frame.GameID.nunique() == games == audit["coverage"]["pbp_games"]
    assert frame.pbp_source_games.eq(1).all()
    assert frame.pbp_pitcher_ids.eq(1).all()
    assert frame.pbp_pitch_count.sum() == audit["pitches"]
    assert frame.pbp_batters_faced.sum() == audit["plate_appearances"]
    assert audit["raw_rows"] - audit["pitches"] == no_pitch == audit["no_pitch_events"]
    assert audit["coverage"]["missing_completed_games_through_pbp_date"] == []
    assert not audit["season_complete"] and not audit["holdout_ready"]
    schedule_bytes = gzip.decompress(path.with_suffix(".schedule.json.gz").read_bytes())
    assert hashlib.sha256(schedule_bytes).hexdigest() == audit["schedule"]["snapshot_sha256"]
    schedule = backfill.schedule_games(json.loads(schedule_bytes), year, date.fromisoformat(audit["schedule"]["through"]))
    completed = set(schedule.loc[schedule.statusCode.eq("RESULT") & ~schedule.cancel, "gameId"])
    assert completed - set(frame.game_pk) == set(audit["coverage"]["missing_completed_games"])
    if year == 2025:
        assert audit["coverage"]["schedule_completed_coverage"]
        # Same four appearances independently checked in MYKBO box scores.
        for name, day, counts in [("장현식", "2025-05-10", [14, 16]), ("박영현", "2025-05-17", [12, 17])]:
            pair = frame.loc[frame.Name.eq(name) & frame.Date.eq(day)]
            assert sorted(pair.pbp_pitch_count) == counts
            assert pair.GameID.nunique() == 2
    else:
        assert frame.Date.max() == "2026-07-26"
        assert not audit["coverage"]["schedule_completed_coverage"]


def test_all_13_legacy_pairs_have_separate_real_pbp_games_without_assigning_ids():
    legacy = pd.read_csv(ROOT / "data/raw/mykbo_2025/2025년_mykbo.csv")
    keys = ["Name", "Date", "Team"]
    duplicates = legacy.loc[legacy.duplicated(keys, keep=False)].copy()
    duplicates["Team"] = duplicates.Team.replace({"Lotte": "롯데", "Kia": "KIA", "Kiwoom": "키움"})
    pbp = pd.read_csv(ROOT / "data/staging/2025/public_pbp_v1.csv")
    ambiguous = []
    for (name, day, team), pair in duplicates.groupby(keys):
        candidates = pbp.loc[pbp.Name.eq(name) & pbp.Date.eq(day) & pbp.Team.eq(team)]
        assert len(candidates) == 2 and candidates.GameID.nunique() == 2
        assert sorted(candidates.pbp_pitch_count) == sorted(pair.NP)
        if pair.NP.nunique() == 1:
            ambiguous.append(name)
    assert ambiguous == ["김영우"]  # Both 15 pitches: NP is not an identity key.
    assert "GameID" not in legacy  # This audit does not migrate the old rows.
