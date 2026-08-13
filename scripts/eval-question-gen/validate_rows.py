#!/usr/bin/env python3
"""Deterministic generated-row validation for Eval-Question-Gen."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from source_dataset import SourceRow, normalize_question


CSV_COLUMNS = [
    "question",
    "answer",
    "docIds",
    "chunk_ids",
    "pipeline",
]


INTERNAL_ID_RE = re.compile(r"\bclf-[a-z0-9]+#\d+\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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


def chunk_doc_id(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else chunk_id


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


def tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value or "") if len(token) > 1}


def jaccard(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def source_rows_by_train_key(rows: list[SourceRow]) -> dict[str, SourceRow]:
    return {row.train_key: row for row in rows}


def validate_one(
    record: dict[str, Any],
    *,
    seen_chunk_ids: set[str],
    source_rows_by_key: dict[str, SourceRow],
    clusters: dict[str, dict[str, Any]],
    exact_training_questions: set[str],
    similarity_threshold: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    for column in CSV_COLUMNS:
        if column not in record:
            reasons.append(f"missing_column:{column}")
    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    doc_ids = record.get("docIds")
    chunk_ids = record.get("chunk_ids")
    if not question:
        reasons.append("empty_question")
    if not answer:
        reasons.append("empty_answer")
    if not isinstance(doc_ids, list) or not all(isinstance(value, str) for value in doc_ids):
        reasons.append("docIds_not_string_array")
        doc_ids = []
    if not isinstance(chunk_ids, list) or not all(isinstance(value, str) for value in chunk_ids):
        reasons.append("chunk_ids_not_string_array")
        chunk_ids = []
    if not chunk_ids:
        reasons.append("empty_chunk_ids")
    unseen_chunks = sorted(set(chunk_ids) - seen_chunk_ids)
    if unseen_chunks:
        reasons.append("unseen_chunk_id")
    chunk_doc_ids = sorted({chunk_doc_id(chunk_id) for chunk_id in chunk_ids})
    if sorted(set(doc_ids)) != chunk_doc_ids:
        reasons.append("docIds_do_not_match_chunk_ids")
    if INTERNAL_ID_RE.search(question):
        reasons.append("question_contains_internal_chunk_id")
    if normalize_question(question) in exact_training_questions:
        reasons.append("exact_training_question_duplicate")

    internal = record.get("internal") or {}
    assignment_allowed_chunk_ids = {
        str(chunk_id)
        for chunk_id in internal.get("allowed_chunk_ids", [])
        if str(chunk_id).strip()
    }
    chunks_outside_assignment = (
        sorted(set(chunk_ids) - assignment_allowed_chunk_ids)
        if assignment_allowed_chunk_ids
        else []
    )
    if chunks_outside_assignment:
        reasons.append("chunk_id_not_in_assignment_allowed_chunks")

    cluster_id = internal.get("cluster_id")
    max_similarity = 0.0
    most_similar_train_key = None
    cluster = clusters.get(str(cluster_id), {})
    if cluster_id and clusters and not cluster:
        reasons.append("unknown_candidate_cluster")
    cluster_chunk_ids = {str(chunk_id) for chunk_id in cluster.get("chunk_ids", [])}
    chunks_outside_cluster = sorted(set(chunk_ids) - cluster_chunk_ids) if cluster_chunk_ids else []
    if chunks_outside_cluster:
        reasons.append("chunk_id_not_in_candidate_cluster")

    related_train_keys = {
        str(value)
        for value in (
            normalize_string_list(cluster.get("seed_train_keys"))
            + normalize_string_list(record.get("seed_train_keys"))
            + normalize_string_list(internal.get("seed_train_keys"))
        )
        if str(value).strip()
    }

    for train_key in sorted(related_train_keys):
        row = source_rows_by_key.get(train_key)
        if not row:
            continue
        similarity = jaccard(question, row.question)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_train_key = row.train_key

    if max_similarity >= similarity_threshold:
        reasons.append("too_similar_to_related_training_question")

    validated = {
        **record,
        "validation": {
            "status": "ok" if not reasons else "reject",
            "reasons": reasons,
            "max_related_training_question_jaccard": round(max_similarity, 4),
            "most_similar_train_key": most_similar_train_key,
            "unseen_chunks": unseen_chunks,
            "chunks_outside_assignment": chunks_outside_assignment,
            "chunks_outside_cluster": chunks_outside_cluster,
            "related_training_record_count": len(related_train_keys),
        },
    }
    return validated


def validate_generated_rows(
    rows: list[SourceRow],
    *,
    run_dir: Path,
    similarity_threshold: float = 0.85,
) -> dict[str, Any]:
    generated = load_jsonl(run_dir / "generated_candidates.jsonl")
    clusters = {
        str(record.get("cluster_id")): record
        for record in load_jsonl(run_dir / "clusters.jsonl")
        if record.get("cluster_id")
    }
    seen_chunk_ids = {
        chunk_id
        for row in rows
        for chunk_id in row.chunk_ids
        if chunk_id
    }
    source_key_map = source_rows_by_train_key(rows)
    exact_training_questions = {
        normalize_question(row.question) for row in rows if row.question
    }

    validated = [
        validate_one(
            record,
            seen_chunk_ids=seen_chunk_ids,
            source_rows_by_key=source_key_map,
            clusters=clusters,
            exact_training_questions=exact_training_questions,
            similarity_threshold=similarity_threshold,
        )
        for record in generated
    ]
    accepted = [
        record for record in validated if record["validation"]["status"] == "ok"
    ]
    rejected = [
        record for record in validated if record["validation"]["status"] != "ok"
    ]

    validated_path = run_dir / "validated_candidates.jsonl"
    accepted_path = run_dir / "validation_accepted_candidates.jsonl"
    rejected_path = run_dir / "validation_rejected_candidates.jsonl"
    summary_path = run_dir / "validation_summary.json"
    write_jsonl(validated, validated_path)
    write_jsonl(accepted, accepted_path)
    write_jsonl(rejected, rejected_path)

    summary = {
        "phase": "validate",
        "generated_count": len(generated),
        "validation_ok_count": len(accepted),
        "validation_reject_count": len(rejected),
        "allowed_chunk_source": "training_set_chunk_ids",
        "related_training_row_source": "candidate_seed_train_keys_union_cluster_seed_train_keys",
        "training_seen_chunk_count": len(seen_chunk_ids),
        "cluster_count": len(clusters),
        "rejection_reason_counts": dict(
            Counter(reason for record in rejected for reason in record["validation"]["reasons"])
        ),
        "artifacts": {
            "validated_candidates": str(validated_path),
            "validation_accepted_candidates": str(accepted_path),
            "validation_rejected_candidates": str(rejected_path),
            "validation_summary": str(summary_path),
        },
        "sample_rejected": rejected[:5],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
