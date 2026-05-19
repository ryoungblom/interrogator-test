"""
Compare multiple result files.

    python -m harness.analyze results_solo_neutral.jsonl results_solo_commission.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .report import print_comparison


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def auto_label(records: list[dict[str, Any]], fallback: str) -> str:
    valid = [r for r in records if "error" not in r]
    if not valid:
        return fallback
    cond = valid[0].get("condition", "?")
    persona = valid[0].get("persona", "?")
    interrog = valid[0].get("interrogator_persona")
    label = f"{cond}_{persona}"
    if interrog and cond == "competitive_with_verifier" and interrog != "honest":
        label += f"_int={interrog}"
    return label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("files", nargs="+", help="One or more JSONL result files.")
    args = ap.parse_args()

    records_by_label: dict[str, list[dict[str, Any]]] = {}
    for p in args.files:
        path = Path(p)
        records = load_jsonl(path)
        label = auto_label(records, fallback=path.stem)
        if label in records_by_label:
            label = f"{label} ({path.stem})"
        records_by_label[label] = records

    print_comparison(records_by_label)


if __name__ == "__main__":
    main()
