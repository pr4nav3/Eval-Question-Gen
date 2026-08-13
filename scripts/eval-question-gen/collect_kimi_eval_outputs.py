#!/usr/bin/env python3
"""Collect Kimi seen-chunk eval outputs into Eval-Question-Gen candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_paths import DEFAULT_ASSIGNMENT_ROOT, DEFAULT_RUN_DIR
from source_dataset import doc_key_for_doc_ids, eval_question_hash


CSV_COLUMNS = [
    "question",
    "answer",
    "docIds",
    "chunk_ids",
    "pipeline",
]
RUN_MEMORY_RELATIVE_PATH = Path("run_memory/generated_eval_rows.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "_parse_error": f"line {line_number}: {exc}",
                        "_raw_line": line.rstrip("\n"),
                    }
                )
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
            else:
                records.append(
                    {
                        "_parse_error": f"line {line_number}: JSON value is not an object",
                        "_raw_line": line.rstrip("\n"),
                    }
                )
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def chunk_doc_id(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else chunk_id


def normalize_candidate(
    row: dict[str, Any],
    *,
    card: dict[str, Any],
    ordinal: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    assignment_id = str(card.get("assignment_id") or "")
    reasons: list[str] = []
    if row.get("_parse_error"):
        return None, {
            "assignment_id": assignment_id,
            "ordinal": ordinal,
            "reasons": ["malformed_jsonl"],
            "error": row.get("_parse_error"),
            "raw_line": row.get("_raw_line"),
        }

    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    chunk_ids = normalize_string_list(row.get("chunk_ids"))
    allowed_chunk_ids = set(normalize_string_list(card.get("allowed_chunk_ids")))
    cited_chunk_ids = sorted(set(chunk_ids), key=chunk_ids.index) if chunk_ids else []
    unauthorized_chunks = sorted(set(cited_chunk_ids) - allowed_chunk_ids)
    if not question:
        reasons.append("empty_question")
    if not answer:
        reasons.append("empty_answer")
    if not cited_chunk_ids:
        reasons.append("empty_chunk_ids")
    if unauthorized_chunks:
        reasons.append("chunk_not_in_assignment")

    derived_doc_ids = sorted({chunk_doc_id(chunk_id) for chunk_id in cited_chunk_ids})
    provided_doc_ids = sorted(set(normalize_string_list(row.get("docIds"))))
    warnings = []
    if not provided_doc_ids:
        reasons.append("empty_docIds")
    elif provided_doc_ids != derived_doc_ids:
        reasons.append("docIds_do_not_match_chunk_ids")

    extra_columns = sorted(set(row) - set(CSV_COLUMNS))
    if extra_columns:
        warnings.append("extra_columns_ignored")

    if reasons:
        return None, {
            "assignment_id": assignment_id,
            "ordinal": ordinal,
            "question": question,
            "chunk_ids": cited_chunk_ids,
            "reasons": reasons,
            "warnings": warnings,
        }

    candidate_id = f"{assignment_id}_q{ordinal:02d}"
    candidate_doc_key = doc_key_for_doc_ids(derived_doc_ids)
    candidate_eval_question_hash = eval_question_hash(question)
    candidate = {
        "question": question,
        "answer": answer,
        "docIds": derived_doc_ids,
        "chunk_ids": cited_chunk_ids,
        "pipeline": "Eval-Question-Gen",
        "eval_question_hash": candidate_eval_question_hash,
        "doc_key": candidate_doc_key,
        "seed_train_keys": card.get("seed_train_keys", []),
        "internal": {
            "candidate_id": candidate_id,
            "assignment_id": assignment_id,
            "cluster_id": card.get("parent_cluster_id") or card.get("cluster_id"),
            "microcluster_id": card.get("cluster_id"),
            "cluster_kind": card.get("cluster_kind"),
            "difficulty": "kimi_selected",
            "question_type": "kimi_selected",
            "allowed_chunk_ids": card.get("allowed_chunk_ids", []),
            "collection_warnings": warnings,
        },
    }
    return candidate, None


def run_memory_record(candidate: dict[str, Any], *, card: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    internal = candidate.get("internal") or {}
    used_chunk_ids = normalize_string_list(candidate.get("chunk_ids"))
    return {
        "created_at": int(time.time()),
        "run_dir": str(run_dir),
        "assignment_id": internal.get("assignment_id") or card.get("assignment_id"),
        "cluster_id": internal.get("cluster_id") or card.get("parent_cluster_id") or card.get("cluster_id"),
        "microcluster_id": internal.get("microcluster_id") or card.get("cluster_id"),
        "cluster_kind": internal.get("cluster_kind") or card.get("cluster_kind"),
        "doc_key": card.get("doc_key") or candidate.get("doc_key") or doc_key_for_doc_ids(normalize_string_list(candidate.get("docIds"))),
        "allowed_docIds": card.get("allowed_docIds", []),
        "used_chunk_ids": used_chunk_ids,
        "question": candidate.get("question", ""),
        "eval_question_hash": candidate.get("eval_question_hash") or eval_question_hash(str(candidate.get("question") or "")),
        "answer": candidate.get("answer", ""),
        "seed_train_keys": candidate.get("seed_train_keys", []),
        "status": "generated",
    }


def dedupe_run_memory_records(
    existing: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {
        (
            str(record.get("assignment_id") or ""),
            str(record.get("eval_question_hash") or eval_question_hash(str(record.get("question") or ""))),
        )
        for record in existing
    }
    deduped: list[dict[str, Any]] = []
    for record in new_records:
        key = (str(record.get("assignment_id") or ""), str(record.get("eval_question_hash") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def write_candidates_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                    "docIds": json.dumps(normalize_string_list(record.get("docIds")), ensure_ascii=False),
                    "chunk_ids": json.dumps(normalize_string_list(record.get("chunk_ids")), ensure_ascii=False),
                    "pipeline": "Eval-Question-Gen",
                }
            )


def collect_outputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_memory_ledger is None:
        args.run_memory_ledger = args.assignment_root / RUN_MEMORY_RELATIVE_PATH
    cards = load_jsonl(args.selected_assignments)
    if not cards:
        raise FileNotFoundError(f"no selected assignments found at {args.selected_assignments}")
    wanted = set(args.assignment_id)
    if wanted:
        cards = [card for card in cards if str(card.get("assignment_id") or "") in wanted]
        missing = sorted(wanted.difference(str(card.get("assignment_id") or "") for card in cards))
        if missing:
            raise ValueError(f"assignment(s) not found: {', '.join(missing)}")

    collected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    run_memory_records: list[dict[str, Any]] = []
    missing_outputs: list[dict[str, str]] = []
    missing_summaries: list[dict[str, str]] = []
    output_file_count = 0
    raw_row_count = 0

    for card in cards:
        output_path = Path(str(card.get("output_path") or ""))
        summary_path = Path(str(card.get("summary_path") or ""))
        assignment_id = str(card.get("assignment_id") or "")
        if not output_path.exists():
            missing_outputs.append({"assignment_id": assignment_id, "output_path": str(output_path)})
            continue
        output_file_count += 1
        raw_rows = load_jsonl(output_path)
        raw_row_count += len(raw_rows)
        for ordinal, row in enumerate(raw_rows, start=1):
            candidate, rejection = normalize_candidate(row, card=card, ordinal=ordinal)
            if candidate:
                collected.append(candidate)
                run_memory_records.append(run_memory_record(candidate, card=card, run_dir=args.run_dir))
            if rejection:
                rejected.append(rejection)
        if args.require_summary and not summary_path.exists():
            missing_summaries.append({"assignment_id": assignment_id, "summary_path": str(summary_path)})

    generated_path = args.run_dir / "generated_candidates.jsonl"
    generated_csv_path = args.run_dir / "generated_candidates.csv"
    rejected_path = args.run_dir / "kimi_collection_rejected.jsonl"
    summary_path = args.run_dir / "kimi_collection_summary.json"
    generation_summary_path = args.run_dir / "generation_summary.json"

    if not args.dry_run:
        write_jsonl(collected, generated_path)
        write_candidates_csv(collected, generated_csv_path)
        write_jsonl(rejected, rejected_path)
        generation_summary = {
            "phase": "generate",
            "source": "kimi_eval_assignments",
            "selected_count": len(cards),
            "generated_count": len(collected),
            "error_count": len(rejected),
            "artifacts": {
                "generated_candidates": str(generated_path),
                "generated_candidates_csv": str(generated_csv_path),
                "generation_errors": str(rejected_path),
                "generation_summary": str(generation_summary_path),
            },
            "sample_generated": collected[:3],
            "sample_errors": rejected[:5],
        }
        generation_summary_path.write_text(
            json.dumps(generation_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        run_memory_written_count = 0
        if args.update_run_memory:
            existing_run_memory = load_jsonl(args.run_memory_ledger)
            to_append = dedupe_run_memory_records(existing_run_memory, run_memory_records)
            append_jsonl(to_append, args.run_memory_ledger)
            run_memory_written_count = len(to_append)
    else:
        run_memory_written_count = 0

    summary = {
        "phase": "collect_kimi",
        "dry_run": args.dry_run,
        "assignment_count": len(cards),
        "output_file_count": output_file_count,
        "missing_output_count": len(missing_outputs),
        "missing_summary_count": len(missing_summaries),
        "raw_row_count": raw_row_count,
        "collected_count": len(collected),
        "rejected_count": len(rejected),
        "rejection_reason_counts": dict(Counter(reason for row in rejected for reason in row.get("reasons", []))),
        "warning_counts": dict(
            Counter(
                warning
                for row in collected
                for warning in (row.get("internal") or {}).get("collection_warnings", [])
            )
        ),
        "run_memory": {
            "ledger": str(args.run_memory_ledger),
            "update_run_memory": args.update_run_memory,
            "would_write_count": len(run_memory_records),
            "written_count": run_memory_written_count,
        },
        "artifacts": {
            "generated_candidates": str(generated_path),
            "generated_candidates_csv": str(generated_csv_path),
            "kimi_collection_rejected": str(rejected_path),
            "kimi_collection_summary": str(summary_path),
            "generation_summary": str(generation_summary_path),
        },
        "sample_missing_outputs": missing_outputs[:5],
        "sample_rejected": rejected[:5],
    }
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--assignment-root", type=Path, default=DEFAULT_ASSIGNMENT_ROOT)
    parser.add_argument("--selected-assignments", type=Path, default=None)
    parser.add_argument(
        "--run-memory-ledger",
        type=Path,
        default=None,
        help="Run-local generated question ledger. Defaults to <assignment-root>/run_memory/generated_eval_rows.jsonl.",
    )
    parser.add_argument("--assignment-id", action="append", default=[])
    parser.add_argument("--require-summary", action="store_true")
    parser.add_argument("--no-update-run-memory", dest="update_run_memory", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(update_run_memory=True)
    args = parser.parse_args()
    if args.selected_assignments is None:
        args.selected_assignments = args.assignment_root / "selected_assignments.jsonl"
    return args


def main() -> int:
    try:
        summary = collect_outputs(parse_args())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
