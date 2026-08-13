#!/usr/bin/env python3
"""Source CSV normalization for Eval-Question-Gen."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = [
    "question",
    "answer",
    "docIds",
    "chunk_ids",
    "pipeline",
    "source_row_number",
    "is_exact_duplicate",
    "duplicate_of_row",
]


@dataclass(frozen=True)
class ParseIssue:
    record_number: int
    csv_line_number: int
    field: str
    value: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_number": self.record_number,
            "csv_line_number": self.csv_line_number,
            "field": self.field,
            "value": self.value,
            "error": self.error,
        }


@dataclass(frozen=True)
class NormalizedChunkRef:
    chunk_id: str
    doc_id: str
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class SourceRow:
    record_number: int
    csv_line_number: int
    question: str
    answer: str
    doc_ids: list[str]
    chunk_ids: list[str]
    normalized_chunks: list[NormalizedChunkRef]
    pipeline: str
    source_row_number: str
    is_exact_duplicate: bool
    duplicate_of_row: str
    raw: dict[str, str]

    @property
    def evidence_signature(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.chunk_ids)))

    @property
    def doc_set_signature(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.doc_ids)))

    @property
    def train_key(self) -> str:
        return train_key_for_row(self.question, self.answer, self.chunk_ids)

    @property
    def doc_key(self) -> str:
        return doc_key_for_doc_ids(self.doc_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_number": self.record_number,
            "csv_line_number": self.csv_line_number,
            "question": self.question,
            "answer": self.answer,
            "docIds": self.doc_ids,
            "chunk_ids": self.chunk_ids,
            "normalized_chunks": [
                chunk.to_dict() for chunk in self.normalized_chunks
            ],
            "pipeline": self.pipeline,
            "is_exact_duplicate": self.is_exact_duplicate,
            "duplicate_of_row": self.duplicate_of_row,
            "train_key": self.train_key,
            "doc_key": self.doc_key,
            "evidence_signature": list(self.evidence_signature),
            "doc_set_signature": list(self.doc_set_signature),
        }


def parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train_key_for_row(question: str, answer: str, chunk_ids: list[str]) -> str:
    return stable_hash(
        {
            "question": normalize_question(question),
            "answer": normalize_question(answer),
            "chunk_ids": sorted(set(str(chunk_id) for chunk_id in chunk_ids if str(chunk_id).strip())),
        }
    )


def doc_key_for_doc_ids(doc_ids: list[str]) -> str:
    return stable_hash(sorted(set(str(doc_id) for doc_id in doc_ids if str(doc_id).strip())))


def eval_question_hash(question: str) -> str:
    return stable_hash({"eval_question": normalize_question(question)})


def parse_json_string_array(
    value: str,
    *,
    record_number: int,
    csv_line_number: int,
    field: str,
    issues: list[ParseIssue],
) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        issues.append(
            ParseIssue(
                record_number=record_number,
                csv_line_number=csv_line_number,
                field=field,
                value=raw,
                error=f"invalid JSON array: {exc.msg}",
            )
        )
        return []
    if not isinstance(parsed, list):
        issues.append(
            ParseIssue(
                record_number=record_number,
                csv_line_number=csv_line_number,
                field=field,
                value=raw,
                error="expected JSON array",
            )
        )
        return []
    values: list[str] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, str):
            issues.append(
                ParseIssue(
                    record_number=record_number,
                    csv_line_number=csv_line_number,
                    field=field,
                    value=repr(item),
                    error=f"array item {index} is not a string",
                )
            )
            continue
        cleaned = item.strip()
        if cleaned:
            values.append(cleaned)
    return values


def normalize_chunk_id(
    chunk_id: str,
    *,
    record_number: int,
    csv_line_number: int,
    issues: list[ParseIssue],
) -> NormalizedChunkRef | None:
    if "#" not in chunk_id:
        issues.append(
            ParseIssue(
                record_number=record_number,
                csv_line_number=csv_line_number,
                field="chunk_ids",
                value=chunk_id,
                error="chunk ID does not contain '#'",
            )
        )
        return None
    doc_id, index_text = chunk_id.rsplit("#", 1)
    doc_id = doc_id.strip()
    index_text = index_text.strip()
    if not doc_id:
        issues.append(
            ParseIssue(
                record_number=record_number,
                csv_line_number=csv_line_number,
                field="chunk_ids",
                value=chunk_id,
                error="chunk ID has empty doc ID",
            )
        )
        return None
    try:
        chunk_index = int(index_text)
    except ValueError:
        issues.append(
            ParseIssue(
                record_number=record_number,
                csv_line_number=csv_line_number,
                field="chunk_ids",
                value=chunk_id,
                error="chunk index is not an integer",
            )
        )
        return None
    if chunk_index < 0:
        issues.append(
            ParseIssue(
                record_number=record_number,
                csv_line_number=csv_line_number,
                field="chunk_ids",
                value=chunk_id,
                error="chunk index is negative",
            )
        )
        return None
    return NormalizedChunkRef(
        chunk_id=f"{doc_id}#{chunk_index}",
        doc_id=doc_id,
        chunk_index=chunk_index,
    )


def load_source_rows(path: Path) -> tuple[list[SourceRow], list[ParseIssue], list[str]]:
    issues: list[ParseIssue] = []
    rows: list[SourceRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for record_number, raw_row in enumerate(reader, start=1):
            csv_line_number = record_number + 1
            raw = {key: str(raw_row.get(key, "") or "") for key in fieldnames}
            doc_ids = parse_json_string_array(
                raw.get("docIds", ""),
                record_number=record_number,
                csv_line_number=csv_line_number,
                field="docIds",
                issues=issues,
            )
            chunk_ids = parse_json_string_array(
                raw.get("chunk_ids", ""),
                record_number=record_number,
                csv_line_number=csv_line_number,
                field="chunk_ids",
                issues=issues,
            )
            normalized_chunks = [
                chunk
                for chunk_id in chunk_ids
                if (
                    chunk := normalize_chunk_id(
                        chunk_id,
                        record_number=record_number,
                        csv_line_number=csv_line_number,
                        issues=issues,
                    )
                )
                is not None
            ]
            rows.append(
                SourceRow(
                    record_number=record_number,
                    csv_line_number=csv_line_number,
                    question=raw.get("question", "").strip(),
                    answer=raw.get("answer", "").strip(),
                    doc_ids=doc_ids,
                    chunk_ids=[chunk.chunk_id for chunk in normalized_chunks],
                    normalized_chunks=normalized_chunks,
                    pipeline=raw.get("pipeline", "").strip(),
                    source_row_number=raw.get("source_row_number", "").strip(),
                    is_exact_duplicate=parse_bool(
                        raw.get("is_exact_duplicate", "")
                    ),
                    duplicate_of_row=raw.get("duplicate_of_row", "").strip(),
                    raw=raw,
                )
            )
    return rows, issues, fieldnames


def build_indexes(rows: list[SourceRow]) -> dict[str, dict[Any, list[int] | set[str]]]:
    chunk_to_rows: dict[str, list[int]] = defaultdict(list)
    doc_to_rows: dict[str, list[int]] = defaultdict(list)
    doc_to_seen_chunks: dict[str, set[str]] = defaultdict(set)
    evidence_signature_to_rows: dict[tuple[str, ...], list[int]] = defaultdict(list)
    doc_set_signature_to_rows: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for row in rows:
        for chunk_id in set(row.chunk_ids):
            chunk_to_rows[chunk_id].append(row.record_number)
        for doc_id in set(row.doc_ids):
            doc_to_rows[doc_id].append(row.record_number)
        for chunk in row.normalized_chunks:
            doc_to_seen_chunks[chunk.doc_id].add(chunk.chunk_id)
        evidence_signature_to_rows[row.evidence_signature].append(row.record_number)
        doc_set_signature_to_rows[row.doc_set_signature].append(row.record_number)

    return {
        "chunk_to_rows": dict(chunk_to_rows),
        "doc_to_rows": dict(doc_to_rows),
        "doc_to_seen_chunks": dict(doc_to_seen_chunks),
        "evidence_signature_to_rows": dict(evidence_signature_to_rows),
        "doc_set_signature_to_rows": dict(doc_set_signature_to_rows),
    }


def percentile(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "min": 0,
            "p25": 0,
            "median": 0,
            "mean": 0,
            "p75": 0,
            "max": 0,
        }
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p25": round(percentile(sorted_values, 0.25), 2),
        "median": round(statistics.median(sorted_values), 2),
        "mean": round(statistics.mean(sorted_values), 2),
        "p75": round(percentile(sorted_values, 0.75), 2),
        "max": sorted_values[-1],
    }


def counter_rows(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(limit)
    ]


def signature_rows(
    grouped_rows: dict[tuple[str, ...], list[int]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    ranked = sorted(
        grouped_rows.items(),
        key=lambda item: (-len(item[1]), len(item[0]), item[0]),
    )
    return [
        {
            "signature": list(signature),
            "row_count": len(row_numbers),
            "record_numbers": row_numbers[:20],
        }
        for signature, row_numbers in ranked[:limit]
    ]


def create_source_analysis(
    path: Path,
    rows: list[SourceRow],
    issues: list[ParseIssue],
    fieldnames: list[str],
) -> dict[str, Any]:
    indexes = build_indexes(rows)
    non_duplicate_rows = [row for row in rows if not row.is_exact_duplicate]
    duplicate_rows = [row for row in rows if row.is_exact_duplicate]

    pipeline_counter = Counter(row.pipeline or "<blank>" for row in rows)
    duplicate_pipeline_counter = Counter(row.pipeline or "<blank>" for row in duplicate_rows)
    doc_row_counter = Counter(
        doc_id for row in rows for doc_id in set(row.doc_ids)
    )
    chunk_row_counter = Counter(
        chunk_id for row in rows for chunk_id in set(row.chunk_ids)
    )
    normalized_question_counter = Counter(
        normalize_question(row.question) for row in rows if row.question
    )
    duplicate_of_counter = Counter(
        row.duplicate_of_row or "<blank>" for row in rows
    )

    rows_missing_doc_ids = [
        row.record_number for row in rows if not row.doc_ids
    ]
    rows_missing_chunk_ids = [
        row.record_number for row in rows if not row.chunk_ids
    ]
    doc_ids_from_chunks = {
        chunk.doc_id for row in rows for chunk in row.normalized_chunks
    }
    doc_ids_from_docids = {doc_id for row in rows for doc_id in row.doc_ids}
    chunk_docs_missing_from_docids_by_row = [
        {
            "record_number": row.record_number,
            "missing_doc_ids": sorted(
                {chunk.doc_id for chunk in row.normalized_chunks}
                - set(row.doc_ids)
            ),
        }
        for row in rows
        if {chunk.doc_id for chunk in row.normalized_chunks} - set(row.doc_ids)
    ]

    evidence_signature_to_rows = indexes["evidence_signature_to_rows"]
    doc_set_signature_to_rows = indexes["doc_set_signature_to_rows"]

    return {
        "input_path": str(path),
        "columns": {
            "actual": fieldnames,
            "expected": EXPECTED_COLUMNS,
            "missing": [col for col in EXPECTED_COLUMNS if col not in fieldnames],
            "extra": [col for col in fieldnames if col not in EXPECTED_COLUMNS],
        },
        "row_counts": {
            "total": len(rows),
            "non_duplicate": len(non_duplicate_rows),
            "exact_duplicate": len(duplicate_rows),
            "rows_missing_doc_ids": len(rows_missing_doc_ids),
            "rows_missing_chunk_ids": len(rows_missing_chunk_ids),
        },
        "unique_counts": {
            "docs_from_docIds": len(doc_ids_from_docids),
            "docs_from_chunk_ids": len(doc_ids_from_chunks),
            "chunks": len(chunk_row_counter),
            "evidence_signatures": len(evidence_signature_to_rows),
            "doc_set_signatures": len(doc_set_signature_to_rows),
            "normalized_questions": len(normalized_question_counter),
        },
        "distributions": {
            "pipeline": counter_rows(pipeline_counter),
            "exact_duplicate_by_pipeline": counter_rows(duplicate_pipeline_counter),
            "duplicate_of_row": counter_rows(duplicate_of_counter),
            "docIds_per_row": numeric_summary([len(row.doc_ids) for row in rows]),
            "chunk_ids_per_row": numeric_summary([len(row.chunk_ids) for row in rows]),
            "normalized_question_reuse": numeric_summary(
                list(normalized_question_counter.values())
            ),
        },
        "top_docs": counter_rows(doc_row_counter),
        "top_chunks": counter_rows(chunk_row_counter),
        "top_evidence_signatures": signature_rows(
            evidence_signature_to_rows, limit=20
        ),
        "top_doc_set_signatures": signature_rows(
            doc_set_signature_to_rows, limit=20
        ),
        "integrity": {
            "parse_issue_count": len(issues),
            "parse_issue_sample": [issue.to_dict() for issue in issues[:50]],
            "rows_missing_doc_ids": rows_missing_doc_ids[:100],
            "rows_missing_chunk_ids": rows_missing_chunk_ids[:100],
            "docs_seen_only_in_docIds": sorted(doc_ids_from_docids - doc_ids_from_chunks)[:100],
            "docs_seen_only_in_chunk_ids": sorted(doc_ids_from_chunks - doc_ids_from_docids)[:100],
            "chunk_docs_missing_from_docIds_by_row_count": len(
                chunk_docs_missing_from_docids_by_row
            ),
            "chunk_docs_missing_from_docIds_by_row_sample": (
                chunk_docs_missing_from_docids_by_row[:50]
            ),
        },
    }


def write_normalized_rows_jsonl(rows: list[SourceRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def write_source_analysis_json(analysis: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
