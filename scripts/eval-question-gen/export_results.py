#!/usr/bin/env python3
"""Final CSV/JSONL export and run reporting for Eval-Question-Gen."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from source_dataset import doc_key_for_doc_ids, eval_question_hash, normalize_question


JUDGE_VERSION = "cluster_local_v2_run_dedupe"

CSV_COLUMNS = [
    "question",
    "answer",
    "docIds",
    "chunk_ids",
    "pipeline",
]


SUMMARY_FILES = [
    "phase_source_summary.json",
    "cluster_summary.json",
    "generation_summary.json",
    "kimi_collection_summary.json",
    "validation_summary.json",
    "judge_summary.json",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_record(record))


def load_final_candidates(run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    judge_summary_path = run_dir / "judge_summary.json"
    judge_path = run_dir / "judge_accepted_candidates.jsonl"
    if not judge_summary_path.exists():
        raise FileNotFoundError(f"judge summary is required before export: {judge_summary_path}")
    judge_summary = load_json(judge_summary_path)
    if judge_summary.get("judge_version") != JUDGE_VERSION:
        raise ValueError(
            f"unsupported judge version in {judge_summary_path}: "
            f"{judge_summary.get('judge_version')!r}"
        )
    if not judge_path.exists():
        raise FileNotFoundError(f"judge summary exists but accepted file is missing: {judge_path}")
    candidates = load_jsonl(judge_path)
    if not candidates:
        raise ValueError(f"judge accepted file is empty: {judge_path}")
    bad = [
        str((record.get("internal") or {}).get("candidate_id") or record.get("question") or "<unknown>")
        for record in candidates
        if (record.get("judge") or {}).get("status") != "accept"
        or (record.get("judge") or {}).get("judge_version") != JUDGE_VERSION
    ]
    if bad:
        raise ValueError(f"judge accepted file contains non-accepted or incompatible rows: {bad[:5]}")
    return "judge_accepted_candidates", candidates


def dedupe_for_export(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for record in records:
        question_key = normalize_question(str(record.get("question") or ""))
        if not question_key:
            rejected.append({**record, "export_reject_reason": "empty_question"})
            continue
        if question_key in seen_questions:
            rejected.append({**record, "export_reject_reason": "duplicate_final_question"})
            continue
        seen_questions.add(question_key)
        accepted.append(record)
    return accepted, rejected


def eval_bank_record(record: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    internal = record.get("internal") if isinstance(record.get("internal"), dict) else {}
    doc_ids = normalize_string_list(record.get("docIds"))
    question = str(record.get("question") or "").strip()
    return {
        "eval_id": str(record.get("eval_id") or internal.get("candidate_id") or eval_question_hash(question)),
        "question": question,
        "answer": str(record.get("answer") or "").strip(),
        "docIds": doc_ids,
        "chunk_ids": normalize_string_list(record.get("chunk_ids")),
        "pipeline": "Eval-Question-Gen",
        "eval_question_hash": str(record.get("eval_question_hash") or eval_question_hash(question)),
        "doc_key": str(record.get("doc_key") or doc_key_for_doc_ids(doc_ids)),
        "seed_train_keys": normalize_string_list(record.get("seed_train_keys")),
        "run_id": run_dir.name,
        "judge_version": str((record.get("judge") or {}).get("judge_version") or JUDGE_VERSION),
    }


def append_eval_bank(records: list[dict[str, Any]], path: Path, *, run_dir: Path) -> int:
    existing = load_jsonl(path)
    seen = {
        str(record.get("eval_question_hash") or eval_question_hash(str(record.get("question") or "")))
        for record in existing
    }
    to_append: list[dict[str, Any]] = []
    for record in records:
        bank_record = eval_bank_record(record, run_dir=run_dir)
        key = bank_record["eval_question_hash"]
        if key in seen:
            continue
        seen.add(key)
        to_append.append(bank_record)
    if not to_append:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in to_append:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(to_append)


def summary_line(label: str, value: Any) -> str:
    if value is None:
        value = ""
    return f"- {label}: {value}"


def summarize_phase(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return []
    phase = summary.get("phase", "unknown")
    lines = [f"### {phase}"]
    preferred_keys = [
        "target",
        "selected_count",
        "generated_count",
        "validation_ok_count",
        "judge_accepted_count",
        "final_count",
        "error_count",
        "rejected_count",
        "hydrated_doc_count",
        "hydrated_chunk_count",
        "cluster_count",
        "candidate_count",
    ]
    for key in preferred_keys:
        if key in summary:
            lines.append(summary_line(key, summary[key]))
    for key in (
        "row_counts",
        "unique_counts",
        "status_counts",
        "cluster_kind_counts",
        "analysis_status_counts",
        "candidate_difficulty_counts",
        "selected_difficulty_counts",
        "rejection_reason_counts",
        "reject_reason_counts",
    ):
        if key in summary:
            lines.append(summary_line(key, json.dumps(summary[key], ensure_ascii=False)))
    return lines


def render_report(
    *,
    run_dir: Path,
    summaries: dict[str, dict[str, Any]],
    source_name: str,
    final_records: list[dict[str, Any]],
    export_rejected: list[dict[str, Any]],
) -> str:
    doc_counts = Counter(
        doc_id
        for record in final_records
        for doc_id in normalize_string_list(record.get("docIds"))
    )
    chunk_counts = Counter(
        chunk_id
        for record in final_records
        for chunk_id in normalize_string_list(record.get("chunk_ids"))
    )
    difficulty_counts = Counter(
        str((record.get("internal") or {}).get("difficulty") or "unknown")
        for record in final_records
    )
    question_type_counts = Counter(
        str((record.get("internal") or {}).get("question_type") or "unknown")
        for record in final_records
    )
    lines = [
        "# Eval-Question-Gen Run Report",
        "",
        summary_line("run_dir", str(run_dir)),
        summary_line("final_source", source_name),
        summary_line("final_count", len(final_records)),
        summary_line("export_rejected_count", len(export_rejected)),
        "",
        "## Final Distribution",
        summary_line("difficulty_counts", json.dumps(dict(difficulty_counts), ensure_ascii=False)),
        summary_line("question_type_counts", json.dumps(dict(question_type_counts), ensure_ascii=False)),
        summary_line("top_docs", json.dumps(doc_counts.most_common(10), ensure_ascii=False)),
        summary_line("top_chunks", json.dumps(chunk_counts.most_common(10), ensure_ascii=False)),
        "",
        "## Phase Summaries",
    ]
    for filename in SUMMARY_FILES:
        phase_lines = summarize_phase(summaries.get(filename, {}))
        if phase_lines:
            lines.extend(["", *phase_lines])

    if final_records:
        lines.extend(["", "## Sample Accepted Rows"])
        for record in final_records[:5]:
            question = str(record.get("question") or "").replace("\n", " ")
            answer = str(record.get("answer") or "").replace("\n", " ")
            if len(answer) > 240:
                answer = answer[:237].rstrip() + "..."
            lines.append(f"- Q: {question}")
            lines.append(f"  A: {answer}")

    if export_rejected:
        lines.extend(["", "## Sample Export Rejections"])
        for record in export_rejected[:5]:
            lines.append(
                "- "
                + json.dumps(
                    {
                        "candidate_id": (record.get("internal") or {}).get("candidate_id"),
                        "reason": record.get("export_reject_reason"),
                    },
                    ensure_ascii=False,
                )
            )
    lines.append("")
    return "\n".join(lines)


def export_final_eval(*, run_dir: Path, eval_bank_path: Path) -> dict[str, Any]:
    source_name, candidates = load_final_candidates(run_dir)
    final_records, export_rejected = dedupe_for_export(candidates)

    csv_path = run_dir / "eval_seen_chunks.csv"
    jsonl_path = run_dir / "eval_seen_chunks.jsonl"
    rejected_path = run_dir / "export_rejected_candidates.jsonl"
    summary_path = run_dir / "export_summary.json"
    report_path = run_dir / "run_report.md"

    write_csv(final_records, csv_path)
    write_jsonl(final_records, jsonl_path)
    write_jsonl(export_rejected, rejected_path)
    eval_bank_written_count = append_eval_bank(final_records, eval_bank_path, run_dir=run_dir)

    summaries = {filename: load_json(run_dir / filename) for filename in SUMMARY_FILES}
    report = render_report(
        run_dir=run_dir,
        summaries=summaries,
        source_name=source_name,
        final_records=final_records,
        export_rejected=export_rejected,
    )
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report)

    summary = {
        "phase": "export",
        "source": source_name,
        "input_candidate_count": len(candidates),
        "final_count": len(final_records),
        "export_rejected_count": len(export_rejected),
        "eval_bank_written_count": eval_bank_written_count,
        "artifacts": {
            "eval_seen_chunks_csv": str(csv_path),
            "eval_seen_chunks_jsonl": str(jsonl_path),
            "export_rejected_candidates": str(rejected_path),
            "run_report": str(report_path),
            "export_summary": str(summary_path),
            "eval_bank": str(eval_bank_path),
        },
        "export_reject_reason_counts": dict(
            Counter(record.get("export_reject_reason", "unknown") for record in export_rejected)
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
