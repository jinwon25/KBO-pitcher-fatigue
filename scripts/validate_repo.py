"""Repository-level integrity checks used locally and in CI."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue import load_dataset, validate_dataset


def validate_markdown_links() -> None:
    pattern = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")
    for markdown in (ROOT / "README.md", ROOT / "docs" / "METHODOLOGY.md"):
        for target in pattern.findall(markdown.read_text(encoding="utf-8")):
            clean = target.split("#", 1)[0]
            if clean and not (markdown.parent / clean).resolve().exists():
                raise AssertionError(f"깨진 링크: {markdown.relative_to(ROOT)} -> {target}")


def validate_notebook() -> None:
    path = ROOT / "notebooks" / "analysis.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert any(cell.get("cell_type") == "code" for cell in notebook["cells"])
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise AssertionError(f"대표 노트북에 저장된 오류가 있습니다: {output}")


def main() -> None:
    frame = load_dataset(ROOT / "data" / "final" / "fatigue_with_index.csv")
    checks = validate_dataset(frame)
    validate_markdown_links()
    validate_notebook()
    print(
        f"Validated {checks['rows']:,} rows, {checks['players']} players, "
        f"{checks['date_min']}–{checks['date_max']}"
    )


if __name__ == "__main__":
    main()
