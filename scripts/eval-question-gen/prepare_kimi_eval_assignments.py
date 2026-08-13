#!/usr/bin/env python3
"""Prepare small Kimi assignments for seen-chunk eval generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_paths import (  # noqa: E402
    DEFAULT_ASSIGNMENT_ROOT,
    DEFAULT_ENV_FILE,
    DEFAULT_INPUT,
    DEFAULT_RUN_DIR,
)
from source_dataset import SourceRow, doc_key_for_doc_ids, load_source_rows  # noqa: E402
from llm_client import apply_env_file  # noqa: E402
from vespa_chunks import (  # noqa: E402
    DEFAULT_VESPA_QUERY_URL,
    DEFAULT_VESPA_RETRIES,
    DEFAULT_VESPA_RETRY_BACKOFF_SECONDS,
    RequestedDoc,
    hydrate_requested_doc,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_ASSIGNMENT_ROOT
DEFAULT_INSTRUCTIONS = SCRIPT_DIR / "KIMI_EVAL_AGENT_INSTRUCTIONS_V3.md"
DEFAULT_MAX_RELATED_TRAINING_ROWS = 200
RUN_MEMORY_RELATIVE_PATH = Path("run_memory/generated_eval_rows.jsonl")
FATAL_HYDRATION_STATUSES = {
    "vespa_access_blocked",
    "vespa_unreachable",
    "vespa_fetch_error",
    "vespa_wrong_doc",
    "vespa_doc_missing",
    "missing_chunks",
    "hydration_incomplete",
}
CSV_COLUMNS = [
    "question",
    "answer",
    "docIds",
    "chunk_ids",
    "pipeline",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
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


def rows_by_train_key(rows: list[SourceRow]) -> dict[str, SourceRow]:
    return {row.train_key: row for row in rows if row.train_key}


def chunk_sort_key(chunk_id: str) -> tuple[str, int, str]:
    if "#" not in chunk_id:
        return (chunk_id, -1, chunk_id)
    doc_id, raw_index = chunk_id.rsplit("#", 1)
    try:
        index = int(raw_index)
    except ValueError:
        index = -1
    return (doc_id, index, chunk_id)


def chunk_doc_id(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else chunk_id


def related_training_rows(
    train_keys: list[str],
    source_rows_by_key: dict[str, SourceRow],
) -> list[dict[str, Any]]:
    candidate_rows = [
        row
        for train_key in sorted(set(str(value) for value in train_keys if str(value).strip()))
        if (row := source_rows_by_key.get(train_key))
    ]
    rows = []
    for row in candidate_rows:
        rows.append(
            {
                "train_key": row.train_key,
                "doc_key": row.doc_key,
                "pipeline": row.pipeline,
                "question": row.question,
                "answer": row.answer,
                "docIds": row.doc_ids,
                "chunk_ids": row.chunk_ids,
            }
        )
    return rows


def ranked_related_training_rows(
    train_keys: list[str],
    *,
    source_rows_by_key: dict[str, SourceRow],
    allowed_chunk_ids: list[str],
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = set(allowed_chunk_ids)
    all_rows = [
        row
        for train_key in sorted(set(str(value) for value in train_keys if str(value).strip()))
        if (row := source_rows_by_key.get(train_key))
    ]

    def sort_key(row: SourceRow) -> tuple[int, int, int, float, int, str]:
        row_chunks = set(row.chunk_ids)
        overlap = row_chunks & allowed
        coverage = len(overlap) / max(1, len(allowed))
        extra_chunks = len(row_chunks - allowed)
        return (
            0 if row_chunks == allowed else 1,
            0 if row_chunks and row_chunks.issubset(allowed) else 1,
            -len(overlap),
            -coverage,
            extra_chunks,
            row.train_key,
        )

    ranked = sorted(all_rows, key=sort_key)
    visible_limit = max(0, max_rows)
    selected = ranked[:visible_limit] if visible_limit else []
    selected_records: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        row_chunks = set(row.chunk_ids)
        selected_records.append(
            {
                "train_key": row.train_key,
                "doc_key": row.doc_key,
                "pipeline": row.pipeline,
                "question": row.question,
                "answer": row.answer,
                "docIds": row.doc_ids,
                "chunk_ids": row.chunk_ids,
                "selection_rank": rank,
                "overlap_chunk_ids": sorted(row_chunks & allowed, key=chunk_sort_key),
                "exact_same_chunk_set": row_chunks == allowed,
                "chunk_set_is_subset_of_assignment": bool(row_chunks) and row_chunks.issubset(allowed),
            }
        )
    stats = {
        "complete_related_record_count": len(all_rows),
        "artifact_row_count": len(selected_records),
        "visible_limit": visible_limit,
        "capped": len(all_rows) > len(selected_records),
        "omitted_count": max(0, len(all_rows) - len(selected_records)),
        "selection_rule": (
            "exact same chunk set, subset of assignment chunks, higher overlap, "
            "higher assignment coverage, fewer extra chunks, then train key"
        ),
    }
    return selected_records, stats


def train_keys_by_chunk_id(rows: list[SourceRow]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for row in rows:
        for chunk_id in row.chunk_ids:
            index.setdefault(chunk_id, []).append(row.train_key)
    return index


def related_train_keys(
    chunk_ids: list[str],
    *,
    train_keys_by_chunk: dict[str, list[str]],
    fallback_train_keys: list[str],
) -> list[str]:
    train_keys: set[str] = {
        str(value).strip()
        for value in fallback_train_keys
        if str(value).strip()
    }
    for chunk_id in chunk_ids:
        train_keys.update(train_keys_by_chunk.get(chunk_id, []))
    return sorted(train_keys)


def load_run_memory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def compact_run_previous_questions(
    *,
    allowed_doc_ids: list[str],
    memory_records: list[dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    current_doc_key = doc_key_for_doc_ids(allowed_doc_ids)
    matches: list[dict[str, Any]] = []
    for record in memory_records:
        status = str(record.get("status") or "")
        if status not in {"generated", "accepted"}:
            continue
        if str(record.get("doc_key") or "") != current_doc_key:
            continue
        matches.append(record)

    def created_at_sort_value(record: dict[str, Any]) -> int:
        try:
            return int(record.get("created_at") or 0)
        except (TypeError, ValueError):
            return 0

    matches.sort(key=lambda record: -created_at_sort_value(record))
    compact: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for record in matches:
        question = str(record.get("question") or "").strip()
        if not question or question.casefold() in seen_questions:
            continue
        seen_questions.add(question.casefold())
        answer = str(record.get("answer") or "").strip()
        compact.append(
            {
                "question": question,
                "answer_short": answer[:500],
                "used_chunk_ids": record.get("used_chunk_ids", []),
                "assignment_id": record.get("assignment_id", ""),
            }
        )
        if len(compact) >= max_rows:
            break
    return compact


def requested_docs_for_chunk_ids(
    chunk_ids: list[str],
    *,
    train_keys_by_chunk: dict[str, list[str]],
) -> list[RequestedDoc]:
    doc_to_indices: dict[str, set[int]] = {}
    for chunk_id in chunk_ids:
        doc_id = chunk_doc_id(chunk_id)
        _, index, _ = chunk_sort_key(chunk_id)
        doc_to_indices.setdefault(doc_id, set()).add(index)

    requested: list[RequestedDoc] = []
    for doc_id in sorted(doc_to_indices):
        indices = sorted(doc_to_indices[doc_id])
        requested.append(
            RequestedDoc(
                doc_id=doc_id,
                chunk_indices=indices,
                chunk_ids=[f"{doc_id}#{index}" for index in indices],
            )
        )
    return requested


def write_assignment_evidence(
    *,
    card: dict[str, Any],
    hydrated_chunks: dict[str, dict[str, Any]],
    train_keys_by_chunk: dict[str, list[str]],
    vespa_query_url: str,
    timeout_seconds: int,
    retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assignment_id = str(card["assignment_id"])
    evidence_chunks_path = Path(card["artifact_paths"]["assignment_hydrated_chunks_jsonl"])
    evidence_docs_path = Path(card["artifact_paths"]["assignment_hydrated_docs_jsonl"])
    allowed_chunk_ids = list(card.get("allowed_chunk_ids", []))
    allowed_set = set(allowed_chunk_ids)
    chunk_records_by_id = {
        chunk_id: hydrated_chunks[chunk_id]
        for chunk_id in allowed_chunk_ids
        if chunk_id in hydrated_chunks
    }
    doc_records: list[dict[str, Any]] = []
    attempted_chunk_records: list[dict[str, Any]] = []

    missing_before_fetch = [
        chunk_id for chunk_id in allowed_chunk_ids if chunk_id not in chunk_records_by_id
    ]
    if missing_before_fetch:
        for requested in requested_docs_for_chunk_ids(
            missing_before_fetch,
            train_keys_by_chunk=train_keys_by_chunk,
        ):
            doc_record, per_doc_chunks = hydrate_requested_doc(
                requested,
                vespa_query_url=vespa_query_url,
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            doc_records.append(doc_record)
            attempted_chunk_records.extend(per_doc_chunks)
            for record in per_doc_chunks:
                chunk_id = str(record.get("chunk_id") or "")
                if chunk_id in allowed_set and record.get("hydration_status") == "ok":
                    chunk_records_by_id[chunk_id] = record

    evidence_records = [
        chunk_records_by_id[chunk_id]
        for chunk_id in allowed_chunk_ids
        if chunk_id in chunk_records_by_id
    ]
    missing_after_fetch = [
        chunk_id for chunk_id in allowed_chunk_ids if chunk_id not in chunk_records_by_id
    ]
    evidence_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_docs_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(evidence_records, evidence_chunks_path)
    write_jsonl(doc_records, evidence_docs_path)

    doc_metadata_by_id: dict[str, dict[str, Any]] = {}
    chunk_refs: list[dict[str, Any]] = []
    for chunk_id in allowed_chunk_ids:
        chunk = chunk_records_by_id.get(chunk_id) or {}
        doc_id = str(chunk.get("doc_id") or chunk_doc_id(chunk_id))
        doc = chunk.get("doc") or {}
        if doc_id and doc and doc_id not in doc_metadata_by_id:
            doc_metadata_by_id[doc_id] = doc
        chunk_refs.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk_sort_key(chunk_id)[1],
                "pages": chunk.get("pages", []),
                "labels": chunk.get("labels", []),
                "text_char_count": len(str(chunk.get("text") or "")),
            }
        )

    doc_status_counts = Counter(str(record.get("status") or "unknown") for record in doc_records)
    chunk_status_counts = Counter(
        str(record.get("hydration_status") or "unknown") for record in attempted_chunk_records
    )
    fetch_error_kind_counts = Counter(
        str(record.get("error_kind") or "none")
        for record in doc_records
        if record.get("status") != "ok"
    )
    failed_doc_records = [
        {
            "doc_id": record.get("doc_id"),
            "status": record.get("status"),
            "error_kind": record.get("error_kind"),
            "error": record.get("error"),
            "attempt_count": record.get("attempt_count"),
            "attempt_errors": record.get("attempt_errors", [])[:3],
        }
        for record in doc_records
        if record.get("status") != "ok"
    ]
    hydration_status = assignment_hydration_status(
        missing_chunk_ids=missing_after_fetch,
        doc_records=doc_records,
        attempted_chunk_records=attempted_chunk_records,
    )
    updated = {
        **card,
        "allowed_chunk_refs": chunk_refs,
        "documents": list(doc_metadata_by_id.values()),
        "missing_chunk_ids": missing_after_fetch,
        "assignment_hydration": {
            "status": hydration_status,
            "requested_chunk_count": len(allowed_chunk_ids),
            "hydrated_chunk_count": len(evidence_records),
            "missing_chunk_count": len(missing_after_fetch),
            "vespa_doc_fetch_count": len(doc_records),
            "vespa_attempt_count_total": sum(
                int(record.get("attempt_count") or 0) for record in doc_records
            ),
            "doc_status_counts": dict(doc_status_counts),
            "chunk_hydration_status_counts": dict(chunk_status_counts),
            "fetch_error_kind_counts": dict(fetch_error_kind_counts),
            "failed_doc_records": failed_doc_records,
        },
    }
    hydration_summary = {
        "assignment_id": assignment_id,
        "status": updated["assignment_hydration"]["status"],
        "requested_chunk_count": len(allowed_chunk_ids),
        "hydrated_chunk_count": len(evidence_records),
        "missing_chunk_ids": missing_after_fetch,
        "vespa_doc_fetch_count": len(doc_records),
        "vespa_attempt_count_total": updated["assignment_hydration"]["vespa_attempt_count_total"],
        "doc_status_counts": dict(doc_status_counts),
        "chunk_hydration_status_counts": dict(chunk_status_counts),
        "fetch_error_kind_counts": dict(fetch_error_kind_counts),
        "failed_doc_records": failed_doc_records,
    }
    return updated, hydration_summary


def assignment_hydration_status(
    *,
    missing_chunk_ids: list[str],
    doc_records: list[dict[str, Any]],
    attempted_chunk_records: list[dict[str, Any]],
) -> str:
    if not missing_chunk_ids:
        return "ok"

    error_kinds = {
        str(record.get("error_kind") or "")
        for record in doc_records
        if record.get("status") == "error"
    }
    doc_statuses = {str(record.get("status") or "") for record in doc_records}
    chunk_statuses = {
        str(record.get("hydration_status") or "")
        for record in attempted_chunk_records
    }

    if "network_permission_denied" in error_kinds:
        return "vespa_access_blocked"
    if error_kinds & {"connection_refused", "timeout"}:
        return "vespa_unreachable"
    if "error" in doc_statuses:
        return "vespa_fetch_error"
    if "wrong_doc" in doc_statuses:
        return "vespa_wrong_doc"
    if "no_hit" in doc_statuses:
        return "vespa_doc_missing"
    if "missing_chunk" in chunk_statuses:
        return "missing_chunks"
    return "hydration_incomplete"


def split_cluster(cluster: dict[str, Any], *, max_chunks: int) -> list[dict[str, Any]]:
    chunk_ids = sorted(set(cluster.get("chunk_ids", [])), key=chunk_sort_key)
    if len(chunk_ids) <= max_chunks:
        return [{**cluster, "microcluster_index": 1, "chunk_ids": chunk_ids}]

    splits: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(chunk_ids), max_chunks), start=1):
        split_chunks = chunk_ids[start : start + max_chunks]
        splits.append(
            {
                **cluster,
                "microcluster_index": index,
                "parent_cluster_id": cluster.get("cluster_id"),
                "cluster_id": f"{cluster.get('cluster_id')}_part{index:02d}",
                "chunk_ids": split_chunks,
                "doc_ids": sorted({chunk_doc_id(chunk_id) for chunk_id in split_chunks}),
                "split_from": cluster.get("split_from") or "kimi_microcluster_budget",
            }
        )
    return splits


def assignment_prompt(card: dict[str, Any]) -> str:
    artifact_paths = card.get("artifact_paths") or {}
    related = card.get("related_training_rows") or {}
    if card.get("question_count_policy") == "natural":
        question_count_line = (
            "- If coherent, produce only as many strong eval rows as this "
            "cluster naturally supports. Prefer 1-3 strong rows and write zero "
            "if the cluster is saturated or weak."
        )
    else:
        question_count_line = f"- If coherent, produce up to {card['target_question_count']} eval rows."
    return "\n".join(
        [
            "# Kimi Seen-Chunk Eval Assignment",
            "",
            "Read these files first:",
            f"- {card['assignment_path']}",
            f"- {card['agent_instructions']}",
            f"- {artifact_paths.get('related_training_rows_jsonl')}",
            "",
            "Task:",
            "- Treat this as a candidate evidence cluster, not guaranteed truth.",
            "- First decide whether the allowed chunks are coherent enough for eval generation.",
            question_count_line,
            "- If weak/noisy/fragmentary, write zero rows and explain why in the summary.",
            "- Read every visible row in the related training rows JSONL before gap finding.",
            (
                "- If the row count appears mismatched against "
                f"`related_training_rows.artifact_row_count` ({related.get('artifact_row_count', 'unknown')}), "
                "note that once in the summary and continue if the evidence is usable."
            ),
            "- Read chunk text from the assignment-local artifact paths only when needed.",
            "- Compare against related training rows, avoid repeating their intent, and target real gaps in question reasoning.",
            "- Do not duplicate questions already written earlier in this assignment run.",
            (
                "- If `run_previous_questions_to_avoid` is non-empty, use it only for "
                "same-run dedupe, not as evidence."
            ),
            "- Append each accepted row immediately to `output_path`.",
            "- Write `summary_path` when done.",
            "",
            "Important:",
            "- Cite only `allowed_chunk_ids` from the assignment.",
            "- Read only the assignment-local hydrated evidence bundle.",
            "- Do not use neighboring chunks.",
            "- Do not search for more evidence.",
            "- Do not mention chunk IDs or assignment IDs in the question.",
            "- Output rows must match the exact CSV fields listed in the assignment.",
            "- Summary should only include reasoning gaps used, packet status, and any one-line issue note.",
            "",
        ]
    )


def prompt_chars(card: dict[str, Any]) -> int:
    return len(assignment_prompt(card)) + len(json.dumps(card, ensure_ascii=False, indent=2))


def make_card(
    *,
    assignment_id: str,
    cluster: dict[str, Any],
    source_rows_by_key: dict[str, SourceRow],
    train_keys_by_chunk: dict[str, list[str]],
    output_root: Path,
    target_question_count: int | None,
    question_count_policy: str,
    run_memory_records: list[dict[str, Any]],
    max_run_previous_questions: int,
    agent_instructions_path: Path | None = None,
) -> dict[str, Any]:
    assignment_path = output_root / "assignments" / f"{assignment_id}.json"
    prompt_path = output_root / "prompts" / f"{assignment_id}.md"
    output_path = output_root / "outputs" / f"{assignment_id}.jsonl"
    summary_path = output_root / "summaries" / f"{assignment_id}.md"
    evidence_chunks_path = output_root / "evidence" / f"{assignment_id}_chunks.jsonl"
    evidence_docs_path = output_root / "evidence" / f"{assignment_id}_docs.jsonl"
    related_training_rows_path = output_root / "training_rows" / f"{assignment_id}_training_rows.jsonl"
    allowed_chunk_ids = list(cluster.get("chunk_ids", []))
    allowed_doc_ids = sorted({chunk_doc_id(chunk_id) for chunk_id in allowed_chunk_ids})
    chunk_refs = [
        {
            "chunk_id": chunk_id,
            "doc_id": chunk_doc_id(chunk_id),
            "chunk_index": chunk_sort_key(chunk_id)[1],
            "pages": [],
            "labels": [],
            "text_char_count": 0,
        }
        for chunk_id in allowed_chunk_ids
    ]
    seed_train_keys = related_train_keys(
        allowed_chunk_ids,
        train_keys_by_chunk=train_keys_by_chunk,
        fallback_train_keys=list(cluster.get("seed_train_keys", [])),
    )
    return {
        "assignment_id": assignment_id,
        "assignment_type": "seen_chunk_eval_microcluster",
        "created_at": int(time.time()),
        "parent_cluster_id": cluster.get("parent_cluster_id") or cluster.get("cluster_id"),
        "cluster_id": cluster.get("cluster_id"),
        "cluster_kind": cluster.get("cluster_kind"),
        "microcluster_index": cluster.get("microcluster_index", 1),
        "support": cluster.get("support", {}),
        "target_question_count": target_question_count,
        "question_count_policy": question_count_policy,
        "csv_columns": CSV_COLUMNS,
        "allowed_docIds": allowed_doc_ids,
        "doc_key": doc_key_for_doc_ids(allowed_doc_ids),
        "allowed_chunk_ids": allowed_chunk_ids,
        "allowed_chunk_refs": chunk_refs,
        "seed_train_keys": sorted(set(seed_train_keys)),
        "related_training_rows": {
            "complete_related_record_count": len(set(seed_train_keys)),
            "artifact_row_count": 0,
            "visible_limit": 0,
            "capped": False,
            "omitted_count": 0,
            "ordering": "ranked_by_overlap_relevance",
        },
        "documents": [],
        "missing_chunk_ids": [],
        "assignment_hydration": {
            "status": "pending",
            "requested_chunk_count": len(allowed_chunk_ids),
            "hydrated_chunk_count": 0,
            "missing_chunk_count": 0,
            "vespa_doc_fetch_count": 0,
        },
        "artifact_paths": {
            "related_training_rows_jsonl": str(related_training_rows_path),
            "assignment_hydrated_chunks_jsonl": str(evidence_chunks_path),
            "assignment_hydrated_docs_jsonl": str(evidence_docs_path),
        },
        "run_previous_questions_to_avoid": compact_run_previous_questions(
            allowed_doc_ids=allowed_doc_ids,
            memory_records=run_memory_records,
            max_rows=max_run_previous_questions,
        ),
        "rules": {
            "exact_chunks_only": True,
            "neighbor_chunks_allowed": False,
            "external_knowledge_allowed": False,
            "one_eval_row_per_training_row": False,
            "cross_run_generated_question_history_allowed": False,
            "pipeline_value": "Eval-Question-Gen",
        },
        "agent_instructions": str(agent_instructions_path or DEFAULT_INSTRUCTIONS),
        "assignment_path": str(assignment_path),
        "prompt_path": str(prompt_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
    }


def cluster_priority(record: dict[str, Any]) -> tuple[int, int, int, str]:
    kind_order = {
        "co_citation": 0,
        "doc_set": 1,
        "doc_local_seen_chunks": 2,
        "multi_doc_bridge": 3,
        "exact_evidence_set": 4,
    }
    return (
        kind_order.get(str(record.get("cluster_kind")), 9),
        -len(record.get("doc_ids", [])),
        len(record.get("chunk_ids", [])),
        str(record.get("cluster_id")),
    )


def select_balanced(
    cards: list[dict[str, Any]],
    *,
    max_assignments: int,
    max_per_doc: int,
) -> list[dict[str, Any]]:
    def chunk_set_key(card: dict[str, Any]) -> tuple[str, ...]:
        return tuple(sorted(set(card.get("allowed_chunk_ids", []))))

    def card_sort_key(card: dict[str, Any]) -> tuple[int, int, int, int, str]:
        doc_count = len(card.get("allowed_docIds", []))
        chunk_count = len(card.get("allowed_chunk_ids", []))
        row_count = int((card.get("support") or {}).get("row_count") or 0)
        return (
            0 if doc_count > 1 else 1,
            abs(4 - chunk_count),
            -min(row_count, 8),
            card["prompt_chars"],
            card["assignment_id"],
        )

    buckets: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        buckets.setdefault(str(card.get("cluster_kind") or "unknown"), []).append(card)
    for bucket in buckets.values():
        bucket.sort(key=card_sort_key)

    selected: list[dict[str, Any]] = []
    doc_counts: Counter[str] = Counter()
    seen_evidence: set[tuple[str, ...]] = set()
    bucket_names = sorted(buckets)
    while len(selected) < max_assignments and any(buckets.values()):
        progressed = False
        for name in bucket_names:
            bucket = buckets.get(name) or []
            while bucket:
                card = bucket.pop(0)
                key = chunk_set_key(card)
                if key in seen_evidence:
                    continue
                doc_ids = card.get("allowed_docIds", [])
                if doc_ids and any(doc_counts[doc_id] >= max_per_doc for doc_id in doc_ids):
                    continue
                selected.append(card)
                seen_evidence.add(key)
                for doc_id in doc_ids:
                    doc_counts[doc_id] += 1
                progressed = True
                break
            if len(selected) >= max_assignments:
                break
        if not progressed:
            break
    return selected


def cluster_score(card: dict[str, Any]) -> float:
    """Quality score for coverage-greedy cluster selection."""
    support = card.get("support") or {}
    row_count = int(support.get("row_count") or 0)
    edge_density = float(support.get("edge_density") or 0.0)
    chunk_count = len(card.get("allowed_chunk_ids", []))
    doc_count = len(card.get("allowed_docIds", []))

    if 4 <= chunk_count <= 6:
        chunk_factor = 1.15
    elif 3 <= chunk_count <= 8:
        chunk_factor = 1.0
    elif chunk_count == 2:
        chunk_factor = 0.7
    else:
        chunk_factor = 0.6

    if 2 <= doc_count <= 3:
        doc_factor = 1.2
    elif doc_count > 5:
        doc_factor = 0.8
    else:
        doc_factor = 1.0

    return edge_density * (row_count ** 0.5) * chunk_factor * doc_factor


def select_coverage_greedy(
    cards: list[dict[str, Any]],
    *,
    target_count: int,
    max_per_doc: int,
    source_rows_by_key: dict[str, SourceRow],
    coverage_threshold: float = 0.98,
) -> list[dict[str, Any]]:
    """Select clusters that maximize coverage of distinct seen chunks.

    Unlike row-coverage selection, every selected cluster is valued by how many
    *new* chunks it contributes. Cluster quality (edge density, size) is used only
    as a tiebreaker. This reduces within-run duplicates caused by overlapping
    evidence sets being picked independently.

    Selection happens in three phases:
    1. Kind minimums: guarantee every cluster kind is represented.
    2. Coverage-greedy: pick clusters that cover the most uncovered chunks.
    3. Fill: add highest-scored unused clusters until target_count is reached.
    """
    all_chunks: set[str] = {
        chunk_id
        for card in cards
        for chunk_id in card.get("allowed_chunk_ids", [])
    }
    uncovered = set(all_chunks)
    coverage_cutoff = int(len(all_chunks) * (1 - coverage_threshold))
    kind_pool_counts = Counter(str(c.get("cluster_kind") or "unknown") for c in cards)
    min_per_kind = {
        k: max(3, int(v * target_count / max(1, len(cards))))
        for k, v in kind_pool_counts.items()
    }

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    doc_counts: Counter[str] = Counter()
    remaining: dict[str, dict[str, Any]] = {card["assignment_id"]: card for card in cards}

    def card_chunks(card: dict[str, Any]) -> set[str]:
        return set(card.get("allowed_chunk_ids", []))

    def add_card(card: dict[str, Any]) -> None:
        selected.append(card)
        selected_ids.add(card["assignment_id"])
        uncovered.difference_update(card_chunks(card))
        remaining.pop(card["assignment_id"], None)
        for doc_id in card.get("allowed_docIds", []):
            doc_counts[doc_id] += 1

    def doc_cap_ok(card: dict[str, Any]) -> bool:
        return not any(doc_counts[doc_id] >= max_per_doc for doc_id in card.get("allowed_docIds", []))

    def sort_key(card: dict[str, Any]) -> tuple[int, float]:
        new_coverage = len(card_chunks(card) & uncovered)
        return (-new_coverage, -cluster_score(card))

    kind_order = sorted({str(card.get("cluster_kind") or "unknown") for card in cards})
    for kind in kind_order:
        kind_added = 0
        kind_floor = min_per_kind.get(kind, 3)
        kind_cards = sorted(
            [card for card in remaining.values() if str(card.get("cluster_kind")) == kind],
            key=sort_key,
        )
        for card in kind_cards:
            if kind_added >= kind_floor or len(selected) >= target_count:
                break
            if not doc_cap_ok(card):
                continue
            if not card_chunks(card):
                continue
            add_card(card)
            kind_added += 1

    while len(selected) < target_count and len(uncovered) > coverage_cutoff:
        best_card: dict[str, Any] | None = None
        best_key: tuple[int, float] | None = None

        for card in remaining.values():
            if not doc_cap_ok(card):
                continue
            key = sort_key(card)
            if key[0] == 0:
                continue
            if best_key is None or key < best_key:
                best_key = key
                best_card = card

        if best_card is None:
            break
        add_card(best_card)

    if len(selected) < target_count:
        fill_candidates = sorted(remaining.values(), key=sort_key)
        for card in fill_candidates:
            if len(selected) >= target_count:
                break
            if not doc_cap_ok(card):
                continue
            add_card(card)

    return selected


def numeric_min_max(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "max": 0}
    return {"min": min(values), "max": max(values)}


def write_related_training_rows_bundle(
    card: dict[str, Any],
    *,
    source_rows_by_key: dict[str, SourceRow],
) -> dict[str, Any]:
    path = Path(card["artifact_paths"]["related_training_rows_jsonl"])
    rows, stats = ranked_related_training_rows(
        [str(value) for value in card.get("seed_train_keys", [])],
        source_rows_by_key=source_rows_by_key,
        allowed_chunk_ids=list(card.get("allowed_chunk_ids", [])),
        max_rows=int((card.get("related_training_rows") or {}).get("visible_limit") or 0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, path)
    return stats


def build_assignments(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file:
        apply_env_file(str(args.env_file))
    if args.run_memory_ledger is None:
        args.run_memory_ledger = args.output_root / RUN_MEMORY_RELATIVE_PATH
    rows, _, _ = load_source_rows(args.input)
    source_rows_by_key = rows_by_train_key(rows)
    chunk_to_train_keys = train_keys_by_chunk_id(rows)
    clusters = load_jsonl(args.run_dir / "clusters.jsonl")
    run_memory_records = load_run_memory(args.run_memory_ledger)
    if not clusters:
        raise FileNotFoundError(f"clusters not found under {args.run_dir}")

    if args.eval_ratio > 0:
        needed_assignments = max(1, math.ceil(len(rows) * args.eval_ratio))
    else:
        needed_assignments = args.max_assignments
        if needed_assignments <= 0:
            needed_assignments = math.ceil(args.target_rows / max(1, args.questions_per_assignment))

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ordinal = 0
    for cluster in sorted(clusters, key=cluster_priority):
        for microcluster in split_cluster(cluster, max_chunks=args.max_chunks):
            ordinal += 1
            assignment_id = f"{args.assignment_prefix}_{ordinal:06d}"
            question_count_policy = "natural" if args.natural_question_count else "bounded"
            card = make_card(
                assignment_id=assignment_id,
                cluster=microcluster,
                source_rows_by_key=source_rows_by_key,
                train_keys_by_chunk=chunk_to_train_keys,
                output_root=args.output_root,
                target_question_count=None if args.natural_question_count else args.questions_per_assignment,
                question_count_policy=question_count_policy,
                run_memory_records=run_memory_records,
                max_run_previous_questions=args.max_run_previous_questions,
                agent_instructions_path=args.agent_instructions,
            )
            card["prompt_chars"] = prompt_chars(card)
            reasons: list[str] = []
            if len(card["allowed_chunk_ids"]) > args.max_chunks:
                reasons.append("too_many_chunks")
            if len(card["allowed_chunk_ids"]) < args.min_chunks:
                reasons.append("too_few_chunks")
            if card["prompt_chars"] > args.max_prompt_chars:
                reasons.append("prompt_too_large")
            if reasons:
                rejected.append(
                    {
                        "assignment_id": assignment_id,
                        "cluster_id": card["cluster_id"],
                        "cluster_kind": card["cluster_kind"],
                        "prompt_chars": card["prompt_chars"],
                        "chunk_count": len(card["allowed_chunk_ids"]),
                        "reasons": reasons,
                    }
                )
                continue
            candidates.append(card)

    if args.eval_ratio > 0:
        selected = select_coverage_greedy(
            candidates,
            target_count=needed_assignments,
            max_per_doc=args.max_assignments_per_doc,
            source_rows_by_key=source_rows_by_key,
        )
    else:
        selected = select_balanced(
            candidates,
            max_assignments=needed_assignments,
            max_per_doc=args.max_assignments_per_doc,
        )
    pre_hydration_selected = selected
    hydration_records: list[dict[str, Any]] = []
    hydration_rejected: list[dict[str, Any]] = []
    vespa_query_url = args.vespa_query_url or os.environ.get(
        "VESPA_QUERY_URL",
        DEFAULT_VESPA_QUERY_URL,
    )
    if args.hydrate_selected and not args.dry_run:
        hydrated_selected: list[dict[str, Any]] = []
        for card in pre_hydration_selected:
            hydrated_card, hydration_record = write_assignment_evidence(
                card=card,
                hydrated_chunks={},
                train_keys_by_chunk=chunk_to_train_keys,
                vespa_query_url=vespa_query_url,
                timeout_seconds=args.vespa_timeout_seconds,
                retries=args.vespa_retries,
                retry_backoff_seconds=args.vespa_retry_backoff_seconds,
            )
            hydrated_card["prompt_chars"] = prompt_chars(hydrated_card)
            hydration_records.append(hydration_record)
            reasons: list[str] = []
            hydration_status = str(
                (hydrated_card.get("assignment_hydration") or {}).get("status") or ""
            )
            if hydration_status and hydration_status != "ok":
                reasons.append(f"{hydration_status}_after_assignment_hydration")
            if hydrated_card["prompt_chars"] > args.max_prompt_chars:
                reasons.append("prompt_too_large_after_assignment_hydration")
            if reasons:
                hydration_rejected.append(
                    {
                        "assignment_id": hydrated_card["assignment_id"],
                        "cluster_id": hydrated_card["cluster_id"],
                        "cluster_kind": hydrated_card["cluster_kind"],
                        "prompt_chars": hydrated_card["prompt_chars"],
                        "chunk_count": len(hydrated_card["allowed_chunk_ids"]),
                        "reasons": reasons,
                        "missing_chunk_ids": hydrated_card.get("missing_chunk_ids", []),
                    }
                )
                continue
            hydrated_selected.append(hydrated_card)
        selected = hydrated_selected

    selected_ids = {card["assignment_id"] for card in pre_hydration_selected}
    covered_train_keys = {
        str(key)
        for card in pre_hydration_selected
        for key in card.get("seed_train_keys", [])
    }
    all_source_chunks = {chunk_id for row in rows for chunk_id in row.chunk_ids if chunk_id}
    covered_chunks = {
        chunk_id
        for card in pre_hydration_selected
        for chunk_id in card.get("allowed_chunk_ids", [])
    }
    unselected = [
        {
            "assignment_id": card["assignment_id"],
            "cluster_id": card["cluster_id"],
            "cluster_kind": card["cluster_kind"],
            "prompt_chars": card["prompt_chars"],
            "chunk_count": len(card["allowed_chunk_ids"]),
            "reason": "not_selected",
        }
        for card in candidates
        if card["assignment_id"] not in selected_ids
    ]

    if not args.dry_run:
        for card in selected:
            assignment_path = Path(card["assignment_path"])
            prompt_path = Path(card["prompt_path"])
            card["related_training_rows"]["visible_limit"] = args.max_related_training_rows
            training_bundle_stats = write_related_training_rows_bundle(
                card,
                source_rows_by_key=source_rows_by_key,
            )
            card["related_training_rows"].update(training_bundle_stats)
            card["prompt_chars"] = prompt_chars(card)
            assignment_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            assignment_path.write_text(
                json.dumps(card, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            prompt_path.write_text(assignment_prompt(card), encoding="utf-8")
        write_jsonl(selected, args.output_root / "selected_assignments.jsonl")
        write_jsonl(
            rejected + hydration_rejected + unselected,
            args.output_root / "rejected_assignments.jsonl",
        )
        write_jsonl(hydration_records, args.output_root / "assignment_hydration.jsonl")

    related_counts = [
        int((card.get("related_training_rows") or {}).get("complete_related_record_count") or 0)
        for card in selected
    ]
    artifact_related_counts = [
        int((card.get("related_training_rows") or {}).get("artifact_row_count") or 0)
        for card in selected
    ]
    fatal_hydration_records = [
        record
        for record in hydration_records
        if str(record.get("status") or "") in FATAL_HYDRATION_STATUSES
    ]

    summary = {
        "run_dir": str(args.run_dir),
        "output_root": str(args.output_root),
        "dry_run": args.dry_run,
        "target_rows": args.target_rows,
        "questions_per_assignment": args.questions_per_assignment,
        "needed_assignments": needed_assignments,
        "eval_ratio": args.eval_ratio,
        "coverage": {
            "total_chunk_count": len(all_source_chunks),
            "covered_chunk_count": len(covered_chunks),
            "coverage_ratio": round(
                len(covered_chunks) / max(1, len(all_source_chunks)), 4
            ),
            "training_row_count": len(source_rows_by_key),
            "covered_train_key_count": len(covered_train_keys),
            "training_row_coverage_ratio": round(
                len(covered_train_keys) / max(1, len(source_rows_by_key)), 4
            ),
        },
        "pre_hydration_selected_assignment_count": len(pre_hydration_selected),
        "selected_assignment_count": len(selected),
        "candidate_assignment_count": len(candidates),
        "rejected_assignment_count": len(rejected),
        "hydration_rejected_assignment_count": len(hydration_rejected),
        "unselected_candidate_count": len(unselected),
        "selected_kind_counts": dict(Counter(card["cluster_kind"] for card in selected)),
        "selected_prompt_chars": {
            "min": min((card["prompt_chars"] for card in selected), default=0),
            "max": max((card["prompt_chars"] for card in selected), default=0),
        },
        "related_training_rows": {
            "complete_count": numeric_min_max(related_counts),
            "artifact_count": numeric_min_max(artifact_related_counts),
            "visible_limit": args.max_related_training_rows,
            "capped_assignment_count": sum(
                1 for card in selected if (card.get("related_training_rows") or {}).get("capped")
            ),
        },
        "limits": {
            "min_chunks": args.min_chunks,
            "max_chunks": args.max_chunks,
            "max_chars_per_chunk": args.max_chars_per_chunk,
            "max_prompt_chars": args.max_prompt_chars,
            "max_assignments_per_doc": args.max_assignments_per_doc,
            "max_run_previous_questions": args.max_run_previous_questions,
            "natural_question_count": args.natural_question_count,
            "max_related_training_rows": args.max_related_training_rows,
        },
        "assignment_hydration": {
            "enabled": args.hydrate_selected,
            "vespa_query_url": vespa_query_url if args.hydrate_selected else None,
            "record_count": len(hydration_records),
            "status_counts": dict(Counter(record["status"] for record in hydration_records)),
            "fatal_statuses": sorted(FATAL_HYDRATION_STATUSES),
            "fatal_count": len(fatal_hydration_records),
        },
        "run_memory": {
            "ledger": str(args.run_memory_ledger),
            "record_count": len(run_memory_records),
            "selected_previous_questions": sum(
                len(card.get("run_previous_questions_to_avoid", [])) for card in selected
            ),
        },
        "artifacts": {
            "selected_assignments": str(args.output_root / "selected_assignments.jsonl"),
            "rejected_assignments": str(args.output_root / "rejected_assignments.jsonl"),
            "assignment_hydration": str(args.output_root / "assignment_hydration.jsonl"),
            "summary": str(args.output_root / "assignment_summary.json"),
        },
        "sample_rejected": rejected[:5],
        "sample_fatal_hydration_failures": fatal_hydration_records[:5],
    }
    if args.verbose:
        summary["sample_selected"] = selected[:3]
    else:
        summary["sample_selected"] = [
            {
                "assignment_id": card["assignment_id"],
                "cluster_id": card["cluster_id"],
                "cluster_kind": card["cluster_kind"],
                "prompt_chars": card["prompt_chars"],
                "doc_count": len(card.get("allowed_docIds", [])),
                "chunk_count": len(card.get("allowed_chunk_ids", [])),
                "seed_train_key_count": len(card.get("seed_train_keys", [])),
                "related_training_rows": card.get("related_training_rows", {}),
            }
            for card in selected[:3]
        ]
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "assignment_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--agent-instructions",
        type=Path,
        default=DEFAULT_INSTRUCTIONS,
        help="Path to the agent instructions markdown file used in assignment prompts.",
    )
    parser.add_argument(
        "--run-memory-ledger",
        type=Path,
        default=None,
        help="Run-local generated question ledger. Defaults to <output-root>/run_memory/generated_eval_rows.jsonl.",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--vespa-query-url", default=None)
    parser.add_argument("--vespa-timeout-seconds", type=int, default=60)
    parser.add_argument("--vespa-retries", type=int, default=DEFAULT_VESPA_RETRIES)
    parser.add_argument("--vespa-retry-backoff-seconds", type=float, default=DEFAULT_VESPA_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--assignment-prefix", default="kimi_eval")
    parser.add_argument("--target-rows", type=int, default=200)
    parser.add_argument("--questions-per-assignment", type=int, default=2)
    parser.add_argument(
        "--natural-question-count",
        action="store_true",
        help="Ask Kimi to write only the number of strong rows the cluster naturally supports.",
    )
    parser.add_argument("--max-assignments", type=int, default=0)
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.0,
        help="If > 0, target assignment count is set to ceil(eval_ratio * training_row_count) and overrides --target-rows/--max-assignments. Best used with --questions-per-assignment 1.",
    )
    parser.add_argument("--max-assignments-per-doc", type=int, default=2)
    parser.add_argument("--min-chunks", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=1200,
        help="Deprecated for Kimi manifests; chunk text is read from assignment-local evidence bundles.",
    )
    parser.add_argument("--max-prompt-chars", type=int, default=30000)
    parser.add_argument("--max-related-training-rows", type=int, default=DEFAULT_MAX_RELATED_TRAINING_ROWS)
    parser.add_argument("--max-run-previous-questions", type=int, default=5)
    parser.add_argument("--no-hydrate-selected", dest="hydrate_selected", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(hydrate_selected=True)
    return parser.parse_args()


def main() -> int:
    summary = build_assignments(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fatal_count = int((summary.get("assignment_hydration") or {}).get("fatal_count") or 0)
    return 2 if fatal_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
