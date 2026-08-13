#!/usr/bin/env python3
"""Targeted exact chunk hydration from Vespa for Eval-Question-Gen."""

from __future__ import annotations

import hashlib
import html
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_VESPA_QUERY_URL = "http://localhost:8081/search/"
DEFAULT_VESPA_RETRIES = 4
DEFAULT_VESPA_RETRY_BACKOFF_SECONDS = 1.5
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
VESPA_FIELDS = (
    "docId,title,fileName,document_id,document_date,referenced_ids,pan_ids,"
    "chunks_summary,chunks_map"
)


@dataclass(frozen=True)
class RequestedDoc:
    doc_id: str
    chunk_indices: list[int]
    chunk_ids: list[str]


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_chunk_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"^\[Page \d+(?:-\d+)?\]\s*", "", text)
    text = html.unescape(text)
    return normalize_ws(text)


def chunk_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("chunk") or value.get("text") or "")
    return ""


def coerce_chunks(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [chunk_text(item) for item in raw]


def chunk_metadata(raw: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(raw, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for fallback_index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("chunk_index", fallback_index))
        except (TypeError, ValueError):
            continue
        result[index] = item
    return result


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def exception_error_kind(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, PermissionError):
            return "network_permission_denied"
        if isinstance(reason, ConnectionRefusedError):
            return "connection_refused"
        if isinstance(reason, TimeoutError | socket.timeout):
            return "timeout"
        return "url_error"
    if isinstance(exc, TimeoutError | socket.timeout):
        return "timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json_response"
    return type(exc).__name__


def fetch_vespa_doc(
    doc_id: str,
    *,
    vespa_query_url: str,
    timeout_seconds: int,
    retries: int = DEFAULT_VESPA_RETRIES,
    retry_backoff_seconds: float = DEFAULT_VESPA_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    base_url = vespa_query_url.rstrip("/") + "/"
    safe_id = doc_id.replace("\\", "\\\\").replace('"', '\\"')
    yql = f'select {VESPA_FIELDS} from kb_items where docId contains "{safe_id}" limit 1'
    query = urllib.parse.urlencode({"yql": yql, "hits": "1"})
    url = base_url + "?" + query
    request = urllib.request.Request(url)
    started_at = time.time()
    attempt_errors: list[str] = []
    max_attempts = max(1, retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error = f"HTTPError: HTTP Error {exc.code}: {exc.reason}"
            error_kind = f"http_{exc.code}"
            attempt_errors.append(error)
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= max_attempts:
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error_kind": error_kind,
                    "fields": {},
                    "error": error,
                    "duration_ms": round((time.time() - started_at) * 1000),
                    "attempt_count": attempt,
                    "attempt_errors": attempt_errors,
                }
            time.sleep(retry_backoff_seconds * attempt)
            continue
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_kind = exception_error_kind(exc)
            attempt_errors.append(error)
            if attempt >= max_attempts:
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error_kind": error_kind,
                    "fields": {},
                    "error": error,
                    "duration_ms": round((time.time() - started_at) * 1000),
                    "attempt_count": attempt,
                    "attempt_errors": attempt_errors,
                }
            time.sleep(retry_backoff_seconds * attempt)
            continue

        children = payload.get("root", {}).get("children", [])
        if not children:
            return {
                "doc_id": doc_id,
                "status": "no_hit",
                "error_kind": None,
                "fields": {},
                "error": None,
                "duration_ms": round((time.time() - started_at) * 1000),
                "attempt_count": attempt,
                "attempt_errors": attempt_errors,
            }

        fields = children[0].get("fields") or {}
        returned_doc_id = normalize_ws(fields.get("docId"))
        if returned_doc_id == doc_id:
            return {
                "doc_id": doc_id,
                "status": "ok",
                "error_kind": None,
                "fields": fields,
                "error": None,
                "duration_ms": round((time.time() - started_at) * 1000),
                "attempt_count": attempt,
                "attempt_errors": attempt_errors,
            }
        return {
            "doc_id": doc_id,
            "status": "wrong_doc",
            "error_kind": "wrong_doc",
            "fields": fields,
            "error": f"expected {doc_id}, got {returned_doc_id or '<blank>'}",
            "duration_ms": round((time.time() - started_at) * 1000),
            "attempt_count": attempt,
            "attempt_errors": attempt_errors,
        }

    return {
        "doc_id": doc_id,
        "status": "error",
        "error_kind": "exhausted_retries",
        "fields": {},
        "error": "exhausted Vespa retries",
        "duration_ms": round((time.time() - started_at) * 1000),
        "attempt_count": max_attempts,
        "attempt_errors": attempt_errors,
    }


def doc_metadata(doc_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "title": normalize_ws(fields.get("title")),
        "fileName": normalize_ws(fields.get("fileName")),
        "document_id": normalize_ws(fields.get("document_id")),
        "document_date": normalize_ws(fields.get("document_date")),
        "referenced_ids": fields.get("referenced_ids") or [],
        "pan_ids": fields.get("pan_ids") or [],
    }


def hydrate_requested_doc(
    requested: RequestedDoc,
    *,
    vespa_query_url: str,
    timeout_seconds: int,
    retries: int = DEFAULT_VESPA_RETRIES,
    retry_backoff_seconds: float = DEFAULT_VESPA_RETRY_BACKOFF_SECONDS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fetched = fetch_vespa_doc(
        requested.doc_id,
        vespa_query_url=vespa_query_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    doc_record = {
        "doc_id": requested.doc_id,
        "status": fetched["status"],
        "requested_chunk_count": len(requested.chunk_indices),
        "requested_chunk_indices": requested.chunk_indices,
        "requested_chunk_ids": requested.chunk_ids,
        "duration_ms": fetched.get("duration_ms"),
        "attempt_count": fetched.get("attempt_count"),
        "attempt_errors": fetched.get("attempt_errors", []),
        "error_kind": fetched.get("error_kind"),
        "error": fetched.get("error"),
    }
    fields = fetched.get("fields") or {}
    if fetched["status"] != "ok":
        hydration_status = {
            "error": "vespa_fetch_error",
            "no_hit": "vespa_doc_missing",
            "wrong_doc": "vespa_wrong_doc",
        }.get(str(fetched["status"]), "vespa_fetch_error")
        chunk_records = [
            {
                "chunk_id": f"{requested.doc_id}#{index}",
                "doc_id": requested.doc_id,
                "chunk_index": index,
                "hydration_status": hydration_status,
                "fetch_status": fetched["status"],
                "fetch_error_kind": fetched.get("error_kind"),
                "text": "",
                "text_sha256": "",
                "pages": [],
                "labels": [],
                "doc": doc_metadata(requested.doc_id, {}),
                "error": fetched.get("error") or fetched["status"],
            }
            for index in requested.chunk_indices
        ]
        return doc_record, chunk_records

    chunks = coerce_chunks(fields.get("chunks_summary"))
    metadata_by_index = chunk_metadata(fields.get("chunks_map"))
    doc_record["total_chunks"] = len(chunks)
    doc_record["doc"] = doc_metadata(requested.doc_id, fields)

    chunk_records: list[dict[str, Any]] = []
    for index in requested.chunk_indices:
        chunk_id = f"{requested.doc_id}#{index}"
        if not 0 <= index < len(chunks):
            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": requested.doc_id,
                    "chunk_index": index,
                    "hydration_status": "missing_chunk",
                    "text": "",
                    "text_sha256": "",
                    "pages": [],
                    "labels": [],
                    "doc": doc_metadata(requested.doc_id, fields),
                    "error": f"chunk index outside document bounds (total={len(chunks)})",
                }
            )
            continue
        raw_text = chunks[index]
        text = clean_chunk_text(raw_text)
        metadata = metadata_by_index.get(index, {})
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "doc_id": requested.doc_id,
                "chunk_index": index,
                "hydration_status": "ok",
                "text": text,
                "text_sha256": sha256_text(text),
                "raw_text_sha256": sha256_text(str(raw_text or "")),
                "pages": list(metadata.get("page_numbers") or []),
                "labels": list(metadata.get("block_labels") or []),
                "doc": doc_metadata(requested.doc_id, fields),
                "error": None,
            }
        )
    return doc_record, chunk_records
