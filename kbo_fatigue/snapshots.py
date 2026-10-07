"""Isolated, immutable staging output shared by collectors and season joins."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def staging_csv_path(root: Path, year: int, output: Path) -> Path:
    if year <= 2024:
        raise ValueError("2020–2024 is frozen; stage only seasons after 2024")
    # Resolve output, but do NOT resolve the allowed directory: that would
    # accidentally authorize a staging symlink pointing into data/final.
    allowed = root.resolve() / "data" / "staging" / str(year)
    output = output.resolve()
    if not output.is_relative_to(allowed) or output.suffix != ".csv":
        raise ValueError(f"Output must be a CSV under {allowed}")
    return output


def publish_snapshot(files: dict[Path, bytes]) -> None:
    """Publish complete files without replacement; roll back ordinary failures.

    When present, the manifest must be last and acts as the completion marker.
    A process/machine crash can leave an orphan CSV; consumers require a manifest.
    """
    temporary = []
    published = []
    try:
        for destination, content in files.items():
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Existing snapshots are immutable: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                path = Path(handle.name)
                temporary.append(path)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for source, destination in zip(temporary, files):
            # Atomic create-if-absent. Unlike replace(), a concurrent writer's
            # snapshot can never be overwritten between the existence check and write.
            os.link(source, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink()
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
