"""Interactive MYKBO collector: explicitly select the season in the browser.

Usage: python crawlers/투수_크롤링.py --players players.json --year 2025
players.json is a name -> numeric MYKBO player ID mapping. The browser selection
is checked against every collected Date; output is always a new staging file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue.appearances import MYKBO_COLUMNS, mykbo_pitching_record, validate_appearances
from kbo_fatigue.snapshots import publish_snapshot, staging_csv_path


def collect_player(driver, *, name: str, player_id: str, year: int) -> list[dict]:
    """Read a fully expanded pitching table; never skip stale/malformed rows."""
    tables = []
    for table in driver.find_elements("css selector", "table"):
        headers = [h.text.strip() for h in table.find_elements("css selector", "thead th")]
        if headers == MYKBO_COLUMNS:
            tables.append(table)
    if len(tables) != 1:
        raise ValueError("Expected one pitching game-log table; inspect current site selectors")
    records = []
    for row in tables[0].find_elements("css selector", "tbody tr"):
        cells = row.find_elements("tag name", "td")
        if not cells:
            raise ValueError("Unexpected empty game-log row")
        records.append(mykbo_pitching_record(
            [cell.text.strip() for cell in cells],
            [a.get_attribute("href") for a in cells[0].find_elements("css selector", "a[href]")],
            name=name, player_id=player_id, year=year,
        ))
    validate_appearances(pd.DataFrame(records))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--snapshot", default="pitchers", help="New output filename stem")
    args = parser.parse_args()
    if args.year <= 2024:
        parser.error("2020–2024 is frozen; use a separate historical migration")
    if Path(args.snapshot).name != args.snapshot or args.snapshot in {"", ".", ".."}:
        parser.error("snapshot must be a filename stem")
    output = staging_csv_path(
        ROOT, args.year, ROOT / "data" / "staging" / str(args.year) / f"{args.snapshot}.csv",
    )
    if output.exists():
        raise FileExistsError(output)
    players = json.loads(args.players.read_text(encoding="utf-8"))
    if not isinstance(players, dict) or not players:
        raise ValueError("players.json must contain a nonempty name -> ID mapping")

    # Browser dependencies are optional and loaded only for a live collection.
    import undetected_chromedriver as uc
    from selenium.webdriver.support.ui import WebDriverWait

    records = []
    driver = uc.Chrome()
    try:
        for name, player_id in players.items():
            driver.get(f"https://mykbostats.com/players/{player_id}")
            WebDriverWait(driver, 20).until(lambda d: d.find_elements("css selector", "table"))
            input(f"[{name}] 브라우저에서 {args.year} 투수 경기 로그를 선택하고 엔터: ")
            # The source has changed UI over time. Fail on unexpected behavior;
            # a wrong year or truncated season must not be silently relabeled.
            for _ in range(100):
                buttons = driver.find_elements("css selector", 'a[phx-click="show_all"]')
                if not buttons:
                    break
                buttons[0].click()
                time.sleep(0.5)
            else:
                raise ValueError("Show More did not finish; no snapshot written")
            records.extend(collect_player(
                driver, name=name, player_id=str(player_id), year=args.year,
            ))
        frame = validate_appearances(pd.DataFrame(records))
    finally:
        driver.quit()
    publish_snapshot({output: frame.to_csv(index=False, lineterminator="\n").encode("utf-8")})
    print(f"Saved {len(frame)} appearances to {output}; season completeness not certified")


if __name__ == "__main__":
    main()
