import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from kbo_fatigue.appearances import (
    APPEARANCE_KEY, MYKBO_COLUMNS, merge_appearances, mykbo_game_identity,
    mykbo_pitching_record, validate_appearances,
)
from scripts import stage_season as staging
from scripts.build_pbp_features import aggregate_game_appearances, build_features

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/mykbo_doubleheaders_2025.csv"


@pytest.fixture
def appearances():
    records = []
    for row in pd.read_csv(FIXTURE, dtype=str).to_dict("records"):
        cells = {column: "" for column in MYKBO_COLUMNS}
        cells.update(Date=row["Date"], NP=row["NP"], IP=row["IP"], Role="RP")
        record = mykbo_pitching_record(
            list(cells.values()), [row["GameURL"]], name=row["Name"],
            player_id=row["PlayerID"], year=2025,
        )
        record["Team"] = row["Team"]
        records.append(record)
    return pd.DataFrame(records)


def test_observed_doubleheaders_join_by_game_not_order(appearances):
    right = appearances[APPEARANCE_KEY + ["Date", "NP"]].rename(columns={"NP": "source_np"})
    joined = merge_appearances(appearances.sample(frac=1, random_state=7), right.iloc[::-1])
    assert len(joined) == 4  # A date join would produce eight rows.
    assert joined.NP.eq(joined.source_np).all()
    assert joined.groupby(["PlayerID", "Date"]).GameID.nunique().tolist() == [2, 2]
    assert set(joined.GameID) == {f"mykbo:game:{i}" for i in (13007, 13011, 13023, 13028)}


def test_url_variants_have_one_stable_id():
    expected = ("mykbo:game:13007", "https://mykbostats.com/games/13007")
    for url in ("/games/13007", "/games/13007-LG-vs-Samsung-20250510",
                "https://mykbostats.com/games/13007/?view=pitching#boxscore"):
        assert mykbo_game_identity(url) == expected


@pytest.mark.parametrize("url", ["", "/games", "/players/13007", "/games/0", "/games/13007/x",
                                     "https://example.com/games/13007", "https://mykbostats.com.evil/games/13007"])
def test_non_game_links_are_rejected(url):
    with pytest.raises(ValueError):
        mykbo_game_identity(url)


@pytest.mark.parametrize("links", [[], ["/games/13007", "/games/13011"]])
def test_missing_or_conflicting_date_links_fail(links):
    with pytest.raises(ValueError, match="exactly one"):
        mykbo_pitching_record(["2025-05-10"] + [""] * 15, links, name="장현식", player_id="572", year=2025)


def test_wrong_season_is_not_relabeled():
    with pytest.raises(ValueError, match="season mismatch"):
        mykbo_pitching_record(["2026-05-10"] + [""] * 15, ["/games/13007"],
                              name="장현식", player_id="572", year=2025)


def test_all_13_legacy_duplicate_groups_stay_unresolved():
    raw = pd.read_csv(ROOT / "data/raw/mykbo_2025/2025년_mykbo.csv")
    keys = ["Name", "Date", "Team"]
    duplicates = raw.loc[raw.duplicated(keys, keep=False)]
    assert raw.duplicated(keys).sum() == 13
    assert len(duplicates) == 26
    assert duplicates.groupby(keys).size().eq(2).all()
    assert duplicates.groupby("Date").size().to_dict() == {"2025-05-10": 4, "2025-05-17": 22}
    assert duplicates.groupby(keys).Time.nunique().eq(1).all()
    with pytest.raises(ValueError, match="re-collect"):
        merge_appearances(raw, raw)


@pytest.mark.parametrize("column,value", [("GameID", None), ("PlayerID", ""), ("Date", None),
                                         ("GameID", "13007"), ("GameURL", "/games/999")])
def test_invalid_identity_is_rejected(appearances, column, value):
    appearances.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_appearances(appearances)


def test_duplicate_game_key_and_metadata_conflicts_fail(appearances):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_appearances(pd.concat([appearances, appearances.iloc[:1]]))
    features = appearances[APPEARANCE_KEY + ["Date"]].copy()
    features.loc[0, "Date"] = "2025-05-11"
    with pytest.raises(ValueError, match="Conflicting.*Date"):
        merge_appearances(appearances, features)
    features.loc[1, "GameID"] = features.loc[0, "GameID"]
    features.loc[1, "PlayerID"] = "mykbo:player:9999"
    with pytest.raises(ValueError, match="multiple dates"):
        validate_appearances(features)


def test_no_silent_unmatched_or_overwritten_metrics(appearances):
    features = appearances[APPEARANCE_KEY + ["Date"]]
    with pytest.raises(ValueError, match="Unmatched"):
        merge_appearances(appearances, features.iloc[:1])
    with pytest.raises(ValueError, match="Unmatched"):
        merge_appearances(appearances.iloc[:1], features)
    with pytest.raises(ValueError, match="Overlapping"):
        merge_appearances(appearances, appearances)


def test_cross_source_ids_need_verified_crosswalk(appearances):
    features = appearances[APPEARANCE_KEY + ["Date"]].replace("mykbo:", "pbp:", regex=True)
    with pytest.raises(ValueError, match="Unmatched"):
        merge_appearances(appearances, features)
    features["PlayerID"] = appearances.PlayerID
    with pytest.raises(ValueError, match="source mismatch"):
        validate_appearances(features)


def test_same_name_different_source_players_are_distinct(appearances):
    other = appearances.iloc[:1].copy()
    other["PlayerID"] = "mykbo:player:9999"
    result = validate_appearances(pd.concat([appearances, other]))
    assert len(result) == 5


@pytest.mark.parametrize("year", [2025, 2026])
def test_season_staging_is_reproducible_and_never_promotes(appearances, tmp_path, monkeypatch, year):
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    appearances["Date"] = appearances.Date.str.replace("2025", str(year))
    appearances["Year"] = year
    source = tmp_path / "source.csv"
    appearances.to_csv(source, index=False)
    output = tmp_path / f"data/staging/{year}/snapshot.csv"
    audit = staging.stage_season([source], [], year, output)
    assert audit["rows"] == 4
    assert audit["identity_validated"]
    assert audit["season_complete"] is audit["holdout_ready"] is False
    assert not (tmp_path / "data/final").exists()
    assert json.loads(output.with_suffix(".manifest.json").read_text())["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    second = output.with_name("snapshot-2.csv")
    staging.stage_season([source], [], year, second)
    assert output.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        staging.stage_season([source], [], year, output)
    with pytest.raises(ValueError, match="under"):
        staging.stage_season([source], [], year, tmp_path / "data/final/fatigue_with_index.csv")
    with pytest.raises(ValueError, match="frozen"):
        staging.stage_season([source], [], 2024, output)


def test_staging_validation_failure_does_not_write(appearances, tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    source = tmp_path / "source.csv"
    appearances.to_csv(source, index=False)
    output = tmp_path / "data/staging/2026/wrong-year.csv"
    with pytest.raises(ValueError, match="another season"):
        staging.stage_season([source], [], 2026, output)
    assert not output.exists()
    output = tmp_path / "data/staging/2025/duplicates.csv"
    with pytest.raises(ValueError, match="Duplicate"):
        staging.stage_season([source, source], [], 2025, output)
    assert not output.exists()


def test_frozen_data_bytes_are_preserved():
    hashes = {
        "fatigue_with_index.csv": "d01f63bfd872c0459fe1585bc66e74da7affa0c5188d3c21945b7c2bd7ebc644",
        "legacy_최종.csv": "64232aab20c29bbcc4e2eb52b2562841ab49d742ea23b7fdd5d1dc585db13f2f",
    }
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / "data/final" / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("date", ["2025-05-10", "2025-05-17"])
def test_pbp_metrics_stay_with_their_source_game(date):
    # Synthetic already-prepared pitches: two same-day games, different counts,
    # velocities, entry innings and RE24. Deliberately interleave the inputs.
    rows = []
    for game, count, speed, inning in [(101, 2, 140., 7), (102, 3, 150., 9)]:
        for n in range(count):
            rows.append(dict(game_pk=game, pitcher=9, pitcher_name="테스트", game_date=date,
                팀="LG", release_speed_kmh=speed, is_hard=True, plate_x=0., plate_z=2.,
                sz_bot=1., sz_top=3., pitch_number=n+1, release_pos_x=1., release_pos_z=5.,
                is_csw=False, is_zone=True, is_first_pitch_strike=True, is_high_pressure=True,
                inning=inning, outs_when_up=0, base_state=0, defense_run_margin=0,
                run_expectancy_before=.5, is_close_late=True))
    pitches = pd.DataFrame(rows)
    pa = pd.DataFrame(dict(game_pk=[101, 102], pitcher=[9, 9], re24_allowed=[.1, .9]))
    result = aggregate_game_appearances(pitches, pa).set_index("GameID")
    assert len(result) == 2
    assert result.pbp_source_games.eq(1).all()
    assert result.pbp_pitcher_ids.eq(1).all()
    assert result.pbp_pitch_count.tolist() == [2, 3]
    assert result.pbp_hard_velocity.tolist() == [140., 150.]
    assert result.pbp_entry_inning.tolist() == [7, 9]
    assert result.pbp_re24_allowed.tolist() == [.1, .9]


def test_legacy_pbp_route_rejects_new_seasons(tmp_path):
    from scripts.build_pbp_features import RAW_COLUMNS
    row = {column: None for column in RAW_COLUMNS}
    row.update(game_date="2025-05-17", pitch_number=1)
    source = tmp_path / "pitches.parquet"
    pd.DataFrame([row]).to_parquet(source)
    with pytest.raises(ValueError, match="frozen"):
        build_features([source], ROOT / "data/final/fatigue_with_index.csv")


def test_collector_preserves_date_cell_links_without_browser():
    spec = importlib.util.spec_from_file_location("pitcher_crawler", ROOT / "crawlers/투수_크롤링.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Element:
        def __init__(self, text="", children=None, href=None):
            self.text, self.children, self.href = text, children or {}, href

        def find_elements(self, by, selector):
            return self.children.get((by, selector), [])

        def get_attribute(self, attribute):
            assert attribute == "href"
            return self.href

    rows = []
    for item in pd.read_csv(FIXTURE, dtype=str).iloc[:2].to_dict("records"):
        values = {column: "" for column in MYKBO_COLUMNS}
        values.update(Date=item["Date"], NP=item["NP"], IP=item["IP"], Role="RP")
        cells = [Element(text=v) for v in values.values()]
        cells[0].children[("css selector", "a[href]")] = [Element(href=item["GameURL"])]
        rows.append(Element(children={("tag name", "td"): cells}))
    table = Element(children={
        ("css selector", "thead th"): [Element(text=v) for v in MYKBO_COLUMNS],
        ("css selector", "tbody tr"): rows,
    })
    driver = Element(children={("css selector", "table"): [table]})
    records = module.collect_player(driver, name="장현식", player_id="572", year=2025)
    assert [r["GameID"] for r in records] == ["mykbo:game:13007", "mykbo:game:13011"]
    assert [r["NP"] for r in records] == ["14", "16"]
    rows[0].children[("tag name", "td")][0].children.clear()
    with pytest.raises(ValueError, match="exactly one"):
        module.collect_player(driver, name="장현식", player_id="572", year=2025)


def test_staging_rejects_symlink_into_frozen_data(tmp_path):
    from kbo_fatigue.snapshots import staging_csv_path
    final = tmp_path / "data/final"
    final.mkdir(parents=True)
    staging_dir = tmp_path / "data/staging"
    staging_dir.mkdir()
    (staging_dir / "2025").symlink_to(final, target_is_directory=True)
    with pytest.raises(ValueError, match="under"):
        staging_csv_path(tmp_path, 2025, staging_dir / "2025/new.csv")
    assert list(final.iterdir()) == []


def test_snapshot_publication_failure_rolls_back_own_files(tmp_path, monkeypatch):
    from kbo_fatigue import snapshots
    output = tmp_path / "snapshot.csv"
    manifest = tmp_path / "snapshot.manifest.json"
    original_link = snapshots.os.link

    def interrupt_manifest(source, destination):
        if destination == manifest:
            raise OSError("simulated disk error")
        return original_link(source, destination)

    monkeypatch.setattr(snapshots.os, "link", interrupt_manifest)
    with pytest.raises(OSError, match="disk error"):
        snapshots.publish_snapshot({output: b"complete csv", manifest: b"{}"})
    assert list(tmp_path.iterdir()) == []


def test_concurrent_snapshot_cannot_be_overwritten(tmp_path, monkeypatch):
    from kbo_fatigue import snapshots
    output = tmp_path / "snapshot.csv"
    original_link = snapshots.os.link

    def concurrent_writer(source, destination):
        destination.write_bytes(b"other writer")
        return original_link(source, destination)

    monkeypatch.setattr(snapshots.os, "link", concurrent_writer)
    with pytest.raises(FileExistsError):
        snapshots.publish_snapshot({output: b"our content"})
    assert output.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [output]


def test_manifest_hashes_the_bytes_actually_parsed(appearances, tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    source = tmp_path / "input.csv"
    appearances.to_csv(source, index=False)
    original_bytes = source.read_bytes()
    original_read = pd.read_csv

    def changing_source(*args, **kwargs):
        result = original_read(*args, **kwargs)
        source.write_bytes(b"changed after reading")
        return result

    monkeypatch.setattr(staging.pd, "read_csv", changing_source)
    audit = staging.stage_season([source], [], 2025, tmp_path / "data/staging/2025/test.csv")
    assert audit["inputs"][0]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert source.read_bytes() != original_bytes


@pytest.mark.parametrize("column,value", [("PlayerID", "mykbo:player:unknown"),
                                         ("GameID", "mykbo:game:013007")])
def test_source_numeric_ids_are_required(appearances, column, value):
    appearances.loc[0, column] = value
    with pytest.raises(ValueError, match="numeric source ID"):
        validate_appearances(appearances)


def test_reserved_or_duplicate_columns_cannot_hide_merge_metadata(appearances):
    with pytest.raises(ValueError, match="Reserved"):
        validate_appearances(appearances.assign(Date__right="2025-05-10"))
    with pytest.raises(ValueError, match="Duplicate column"):
        validate_appearances(pd.concat([appearances, appearances[["Date"]]], axis=1))


def test_new_pbp_source_overlap_is_rejected_before_aggregation(tmp_path):
    from scripts.build_pbp_features import RAW_COLUMNS
    row = {column: None for column in RAW_COLUMNS}
    row.update(game_pk=101, pitcher=9, pitcher_name="테스트", game_date="2025-05-17",
               at_bat_number=1, pitch_number=1)
    source = tmp_path / "pitches.parquet"
    pd.DataFrame([row]).to_parquet(source)
    with pytest.raises(ValueError, match="Duplicate PBP pitch"):
        build_features([source, source], None, game_level=True)


@pytest.mark.parametrize("no_pitch_walk", [False, True])
def test_game_level_pbp_parquet_pipeline_separates_doubleheader(tmp_path, no_pitch_walk):
    from scripts.build_pbp_features import RAW_COLUMNS
    rows = []
    for game, speed in [(101, 140.), (102, 150.)]:
        for outs in range(3):
            for bases in range(8):
                row = {column: None for column in RAW_COLUMNS}
                row.update(game_pk=game, pitcher=9, pitcher_name="테스트", game_date="2025-05-17",
                    home_team="LG", away_team="KT", inning=1, inning_topbot="top",
                    at_bat_number=outs*8+bases+1, pitch_number=1, home_score=0, away_score=0,
                    pitch_result="T", type="S", pitch_type="FF", release_speed_kmh=speed,
                    plate_x=0., plate_z=2., sz_bot=1., sz_top=3., outs_when_up=outs,
                    on_1b=11 if bases & 1 else None, on_2b=12 if bases & 2 else None,
                    on_3b=13 if bases & 4 else None, post_home_score=0, post_away_score=0,
                    post_outs=3, runs_scored=0, release_pos_x=1., release_pos_z=5.)
                rows.append(row)
    if no_pitch_walk:
        for game in (101, 102):
            walk = next(row.copy() for row in rows if row["game_pk"] == game)
            walk.update(at_bat_number=0, pitch_number=0, release_speed_kmh=None, on_2b=12,
                        pitch_type=None, pitch_result=None, type=None, post_outs=0, post_on_1b=11)
            rows.append(walk)
    source = tmp_path / "doubleheader.parquet"
    pd.DataFrame(rows).to_parquet(source)
    actual, expectancy = build_features([source], None, game_level=True)
    assert actual.GameID.tolist() == ["pbp:game:101", "pbp:game:102"]
    assert actual.pbp_pitch_count.tolist() == [24, 24]
    assert actual.pbp_batters_faced.tolist() == [24 + no_pitch_walk, 24 + no_pitch_walk]
    assert actual.pbp_hard_velocity.tolist() == [140., 150.]
    assert actual.pbp_entry_runners.tolist() == [int(no_pitch_walk)] * 2
    assert actual.pbp_source_games.eq(1).all()
    assert len(expectancy) == 24


def test_no_pitch_only_appearance_is_retained_with_entry_context():
    row = dict(game_pk=101, pitcher=9, pitcher_name="테스트", game_date="2025-05-17",
        팀="LG", release_speed_kmh=None, is_hard=False, plate_x=None, plate_z=None,
        sz_bot=None, sz_top=None, pitch_number=0, release_pos_x=None, release_pos_z=None,
        is_csw=False, is_zone=False, is_first_pitch_strike=False, is_high_pressure=True,
        inning=8, outs_when_up=1, base_state=6, defense_run_margin=1,
        run_expectancy_before=.5, is_close_late=True)
    pitches = pd.DataFrame([row])
    pa = pd.DataFrame([dict(game_pk=101, pitcher=9, re24_allowed=.2)])
    actual = aggregate_game_appearances(pitches, pa).iloc[0]
    assert actual.pbp_pitch_count == 0 and actual.pbp_batters_faced == 1
    assert actual.pbp_source_games == 1 and actual.pbp_pitcher_ids == 1
    assert actual.pbp_entry_inning == 8 and actual.pbp_entry_runners == 2
    assert actual.pbp_re24_allowed == .2
    assert pd.isna(actual.pbp_avg_velocity) and pd.isna(actual.pbp_hard_usage)


def test_zero_and_thrown_events_in_one_pa_are_quarantined(tmp_path):
    from scripts.build_pbp_features import RAW_COLUMNS
    row = {column: None for column in RAW_COLUMNS}
    row.update(game_pk=101, pitcher=9, pitcher_name="테스트", game_date="2025-05-17",
               at_bat_number=1, pitch_number=1)
    source = tmp_path / "ambiguous.parquet"
    pd.DataFrame([row, dict(row, pitch_number=0)]).to_parquet(source)
    with pytest.raises(ValueError, match="Ambiguous zero-pitch"):
        build_features([source], None, game_level=True)
