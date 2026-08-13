#!/usr/bin/env python3
"""Compute delta train keys for Eval-Question-Gen."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pipeline_paths import DEFAULT_DELTAS_ROOT, DEFAULT_INPUT, DEFAULT_TRAINING_STATE
from source_dataset import load_source_rows


DEFAULT_STATE_PATH = DEFAULT_TRAINING_STATE
DEFAULT_OUTPUT_ROOT = DEFAULT_DELTAS_ROOT


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def state_record(row: Any, *, snapshot_id: str) -> dict[str, str]:
    return {
        "train_key": row.train_key,
        "doc_key": row.doc_key,
        "first_seen_snapshot": snapshot_id,
    }


def run_delta(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_id = args.snapshot_id or time.strftime("train_%Y%m%d_%H%M%S")
    output_dir = args.output_root / snapshot_id
    rows, _, _ = load_source_rows(args.input)

    existing_records = load_jsonl(args.state_path)
    known_train_keys = {str(record.get("train_key") or "") for record in existing_records}
    current_records_by_key = {
        row.train_key: state_record(row, snapshot_id=snapshot_id)
        for row in rows
        if row.train_key
    }
    delta_records = [
        record
        for train_key, record in sorted(current_records_by_key.items())
        if train_key not in known_train_keys
    ]

    delta_keys_path = output_dir / "delta_train_keys.jsonl"
    summary_path = output_dir / "training_delta_summary.json"
    write_jsonl([{"train_key": record["train_key"]} for record in delta_records], delta_keys_path)

    state_written_count = 0
    if args.update_state:
        append_jsonl(delta_records, args.state_path)
        state_written_count = len(delta_records)

    summary = {
        "phase": "training_delta",
        "snapshot_id": snapshot_id,
        "input_path": str(args.input),
        "state_path": str(args.state_path),
        "current_row_count": len(rows),
        "current_unique_train_key_count": len(current_records_by_key),
        "known_train_key_count": len(known_train_keys),
        "new_train_key_count": len(delta_records),
        "new_doc_key_count": len({record["doc_key"] for record in delta_records}),
        "update_state": args.update_state,
        "state_written_count": state_written_count,
        "artifacts": {
            "delta_train_keys": str(delta_keys_path),
            "summary": str(summary_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--update-state", dest="update_state", action="store_true")
    parser.add_argument("--no-update-state", dest="update_state", action="store_false")
    parser.set_defaults(update_state=True)
    return parser.parse_args()


def main() -> int:
    summary = run_delta(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
