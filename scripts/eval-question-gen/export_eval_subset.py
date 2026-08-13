#!/usr/bin/env python3
"""Export eval-bank rows relevant to a training subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pipeline_paths import DEFAULT_EVAL_BANK
from source_dataset import load_source_rows


CSV_COLUMNS = ["question", "answer", "docIds", "chunk_ids", "pipeline"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if parsed:
            return [str(parsed).strip()]
    return []


def csv_record(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question": str(record.get("question") or "").strip(),
        "answer": str(record.get("answer") or "").strip(),
        "docIds": json.dumps(normalize_string_list(record.get("docIds")), ensure_ascii=False),
        "chunk_ids": json.dumps(normalize_string_list(record.get("chunk_ids")), ensure_ascii=False),
        "pipeline": "Eval-Question-Gen",
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_record(record))


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    rows, _, _ = load_source_rows(args.subset_input)
    subset_train_keys = {row.train_key for row in rows if row.train_key}
    subset_chunk_ids = {chunk_id for row in rows for chunk_id in row.chunk_ids if chunk_id}

    selected: list[dict[str, Any]] = []
    rejected_counts = {"no_seed_overlap": 0, "chunk_outside_subset": 0}
    seen_eval_questions: set[str] = set()
    for record in load_jsonl(args.eval_bank):
        eval_hash = str(record.get("eval_question_hash") or "")
        if eval_hash and eval_hash in seen_eval_questions:
            continue
        seed_train_keys = set(normalize_string_list(record.get("seed_train_keys")))
        if not seed_train_keys & subset_train_keys:
            rejected_counts["no_seed_overlap"] += 1
            continue
        chunk_ids = set(normalize_string_list(record.get("chunk_ids")))
        if args.require_chunk_subset and not chunk_ids.issubset(subset_chunk_ids):
            rejected_counts["chunk_outside_subset"] += 1
            continue
        if eval_hash:
            seen_eval_questions.add(eval_hash)
        selected.append(record)
        if args.limit and len(selected) >= args.limit:
            break

    write_csv(selected, args.output_csv)
    summary = {
        "phase": "export_eval_subset",
        "subset_input": str(args.subset_input),
        "eval_bank": str(args.eval_bank),
        "subset_row_count": len(rows),
        "subset_train_key_count": len(subset_train_keys),
        "subset_chunk_count": len(subset_chunk_ids),
        "selected_eval_count": len(selected),
        "require_chunk_subset": args.require_chunk_subset,
        "rejected_counts": rejected_counts,
        "artifacts": {"output_csv": str(args.output_csv)},
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-input", type=Path, required=True)
    parser.add_argument("--eval-bank", type=Path, default=DEFAULT_EVAL_BANK)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-chunk-subset", dest="require_chunk_subset", action="store_true")
    parser.add_argument("--allow-extra-chunks", dest="require_chunk_subset", action="store_false")
    parser.set_defaults(require_chunk_subset=True)
    return parser.parse_args()


def main() -> int:
    summary = run_export(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
