"""Game-aware collection and fail-closed joins for new season staging.

The frozen 2020–2024 dataset has a different, historical date-level contract.
Never manufacture game identity from row order, date, opponent or start time.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

import pandas as pd


APPEARANCE_KEY = ["PlayerID", "GameID"]
MYKBO_COLUMNS = [
    "Date", "Opp", "Role", "Dec", "ERA", "WHIP", "IP", "NP", "R", "ER",
    "H", "HR", "SO", "BB", "HB", "GS",
]


def mykbo_game_identity(href: str) -> tuple[str, str]:
    """Accept numeric game URLs with optional display slugs, queries/fragments."""
    if not isinstance(href, str) or not href.strip():
        raise ValueError(f"Missing MYKBO game URL: {href!r}")
    url = urlsplit(urljoin("https://mykbostats.com/", href.strip()))
    match = re.fullmatch(r"/games/([1-9][0-9]*)(?:-[^/]*)?/?", url.path)
    if (
        url.scheme not in {"http", "https"}
        or url.netloc.lower() not in {"mykbostats.com", "www.mykbostats.com"}
        or not match
    ):
        raise ValueError(f"Not a MYKBO game URL: {href!r}")
    source_id = match.group(1)
    return f"mykbo:game:{source_id}", f"https://mykbostats.com/games/{source_id}"


def mykbo_pitching_record(
    cells: list[str], date_links: list[str], *, name: str, player_id: str, year: int
) -> dict:
    """Convert a rendered game-log row, retaining its date-cell game link.

    Called by the Selenium collector and offline regression tests. Fail on DOM
    drift, a wrong selected season, missing or conflicting game links.
    """
    if len(cells) != len(MYKBO_COLUMNS):
        raise ValueError(f"Expected 16 game-log cells, got {len(cells)}")
    if not re.fullmatch(r"[1-9][0-9]*", str(player_id)):
        raise ValueError("MYKBO player_id must be the numeric source ID")
    if not name.strip():
        raise ValueError("Player name is empty")
    identities = {mykbo_game_identity(link) for link in date_links}
    if len(identities) != 1:
        raise ValueError("Each pitching row requires exactly one game identity")
    record = dict(zip(MYKBO_COLUMNS, (cell.strip() for cell in cells)))
    date = pd.to_datetime(record["Date"], format="%Y-%m-%d", errors="raise")
    if pd.isna(date) or date.year != year:
        raise ValueError(f"Selected season mismatch: expected {year}, got {record['Date']}")
    game_id, game_url = identities.pop()
    record.update(
        Name=name.strip(), PlayerID=f"mykbo:player:{player_id}", Year=year,
        Date=date.date().isoformat(), GameID=game_id, GameURL=game_url,
    )
    return record


def validate_appearances(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy; reject missing, repeated or inconsistent keys."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column labels")
    if any(str(c) == "__match" or str(c).endswith("__right") for c in frame.columns):
        raise ValueError("Reserved merge column name")
    required = APPEARANCE_KEY + ["Date"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing game-aware columns: {sorted(missing)}; re-collect source IDs")
    if frame.empty:
        raise ValueError("No appearances collected")
    result = frame.copy()
    for column in required:
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Null/empty identity field: {column}")
    for column, kind in (("PlayerID", "player"), ("GameID", "game")):
        result[column] = result[column].astype(str).str.strip()
        if not result[column].str.fullmatch(rf"[a-z][a-z0-9_]*:{kind}:[A-Za-z0-9_-]+").all():
            raise ValueError(f"{column} must have an explicit source namespace")
    player_source = result["PlayerID"].str.split(":").str[0]
    game_source = result["GameID"].str.split(":").str[0]
    if not player_source.eq(game_source).all():
        raise ValueError("PlayerID/GameID source mismatch; a verified crosswalk is required")
    for column, kind in (("PlayerID", "player"), ("GameID", "game")):
        mykbo = result.loc[game_source.eq("mykbo"), column]
        if not mykbo.str.fullmatch(rf"mykbo:{kind}:[1-9][0-9]*").all():
            raise ValueError(f"MYKBO {column} must contain a numeric source ID")
    dates = pd.to_datetime(result["Date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Date must be a non-null game date, without time")
    result["Date"] = dates.dt.strftime("%Y-%m-%d")
    if "Year" in result and not pd.to_numeric(result["Year"], errors="raise").eq(dates.dt.year).all():
        raise ValueError("Year disagrees with Date")
    if result.duplicated(APPEARANCE_KEY).any():
        raise ValueError("Duplicate (PlayerID, GameID); do not drop duplicates or use cumcount")
    if result.groupby("GameID")["Date"].nunique().gt(1).any():
        raise ValueError("One GameID maps to multiple dates")
    if "GameURL" in result:
        for game_id, game_url in zip(result["GameID"], result["GameURL"]):
            if game_id.startswith("mykbo:"):
                if pd.isna(game_url) or mykbo_game_identity(str(game_url))[0] != game_id:
                    raise ValueError("GameURL disagrees with GameID")
    return result


def merge_appearances(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Strict one-to-one appearance enrichment; no silent loss or date fallback.

    Both tables must cover the same appearance set. Metadata shared by both
    sides must agree; overlapping metric columns are rejected, not overwritten.
    Cross-source inputs must first use a reviewed game AND player ID crosswalk.
    """
    left, right = validate_appearances(left), validate_appearances(right)
    common = (set(left) & set(right)) - set(APPEARANCE_KEY)
    metadata = {"Date", "Year", "Name", "Team", "Opp", "GameURL"}
    if common - metadata:
        raise ValueError(f"Overlapping metric columns: {sorted(common - metadata)}")
    joined = left.merge(
        right, on=APPEARANCE_KEY, how="outer", validate="one_to_one",
        suffixes=("", "__right"), indicator="__match", sort=False,
    )
    counts = joined["__match"].value_counts()
    if not joined["__match"].eq("both").all():
        raise ValueError(f"Unmatched appearances: {counts.to_dict()}")
    for column in sorted(common):
        a, b = joined[column], joined[f"{column}__right"]
        if column == "GameURL":
            a = a.map(lambda url: mykbo_game_identity(str(url))[0])
            b = b.map(lambda url: mykbo_game_identity(str(url))[0])
        equal = a.eq(b) | (a.isna() & b.isna())
        if not equal.fillna(False).all():
            raise ValueError(f"Conflicting appearance metadata: {column}")
    joined = joined.drop(columns=["__match"] + [f"{c}__right" for c in common])
    return validate_appearances(joined).sort_values(APPEARANCE_KEY).reset_index(drop=True)
