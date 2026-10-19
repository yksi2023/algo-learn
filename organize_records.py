"""Move existing flat records into one directory per experiment suite."""

import argparse
import json
from pathlib import Path


def organize(records: Path) -> list[Path]:
    moves = []
    suites = set()
    for path in sorted(records.glob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        suite = metadata["suite_id"]
        suites.add(suite)
        for suffix in (".json", ".csv", ".pt", ".pt.tmp"):
            source = records / f"{path.stem}{suffix}"
            if source.exists():
                moves.append((source, records / suite / source.name))
    for suite in sorted(suites):
        for name in ("curves.png", "curves.pdf", "summary.csv", "conditions.csv"):
            source = records / f"{suite}-{name}"
            if source.exists():
                moves.append((source, records / suite / name))
    # Check all destinations before moving any data; never overwrite a run.
    for _, destination in moves:
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    return [destination for _, destination in moves]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("records"))
    args = parser.parse_args()
    paths = organize(args.records)
    print(f"Moved {len(paths)} files into suite directories under {args.records}.")


if __name__ == "__main__":
    main()
