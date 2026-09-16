"""Execute the representative notebook with the current Python interpreter."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nbformat
from ipykernel.kernelspec import install as install_kernel
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "analysis.ipynb"


def execute(in_place: bool = False) -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    with tempfile.TemporaryDirectory(prefix="kbo-notebook-") as directory:
        prefix = Path(directory)
        install_kernel(prefix=str(prefix), kernel_name="kbo-fatigue", display_name="KBO fatigue")
        original_path = os.environ.get("JUPYTER_PATH", "")
        kernel_path = str(prefix / "share" / "jupyter")
        os.environ["JUPYTER_PATH"] = kernel_path + (os.pathsep + original_path if original_path else "")
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="kbo-fatigue",
            resources={"metadata": {"path": str(ROOT)}},
        )
        executed = client.execute()

    output = NOTEBOOK if in_place else Path(tempfile.gettempdir()) / "kbo-fatigue-analysis.ipynb"
    nbformat.write(executed, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", help="Save outputs in the tracked notebook")
    args = parser.parse_args()
    output = execute(in_place=args.in_place)
    print(f"Executed notebook: {output}")


if __name__ == "__main__":
    main()
