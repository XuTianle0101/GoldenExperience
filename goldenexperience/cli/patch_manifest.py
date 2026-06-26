"""Print the default GoldenExperience LMCache patch manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from goldenexperience.lmcache_patch import PatchManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the GoldenExperience LMCache patch manifest.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write markdown.")
    args = parser.parse_args()
    manifest = PatchManifest.default().as_markdown()
    if args.output is None:
        print(manifest, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
