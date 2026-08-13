#!/usr/bin/env python3
"""Hands-free Eval-Question-Gen supervisor.

This supervises already prepared Kimi eval assignments. It keeps a single global
LLM-agent budget across generation workers and assignment-scoped judge workers.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_paths import (  # noqa: E402
    DEFAULT_ASSIGNMENT_ROOT,
    DEFAULT_ENV_FILE,
    DEFAULT_EVAL_BANK,
    DEFAULT_INPUT,
    DEFAULT_RUN_DIR,
)
from collect_kimi_eval_outputs import (  # noqa: E402
    append_jsonl as append_collect_jsonl,
    dedupe_run_memory_records,
    load_jsonl as collect_load_jsonl,
    normalize_candidate,
    run_memory_record,
    write_candidates_csv,
    write_jsonl as write_collect_jsonl,
)
from export_results import export_final_eval  # noqa: E402
from judge_eval_rows import JUDGE_VERSION  # noqa: E402
from run_kimi_eval_assignments import (  # noqa: E402
    DEFAULT_WORKER_COMMAND,
    command_for_assignment,
    nonblank_line_count,
    worker_env,
)
from source_dataset import load_source_rows, normalize_question  # noqa: E402
from validate_rows import (  # noqa: E402
    source_rows_by_train_key,
    validate_one,
)


DEFAULT_LLM_MODEL = "private-large"
STATE_SCHEMA_VERSION = 1


def now_ts() -> int:
    return int(time.time())


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_event(root: Path, event: str, **payload: Any) -> None:
    path = root / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": now_ts(), "time": iso_now(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assignment_id(card: dict[str, Any]) -> str:
    return str(card.get("assignment_id") or "").strip()


def candidate_id(record: dict[str, Any]) -> str:
    internal = record.get("internal") if isinstance(record.get("internal"), dict) else {}
    return str(internal.get("candidate_id") or record.get("eval_question_hash") or record.get("question") or "")


def supervisor_root(run_dir: Path) -> Path:
    return run_dir / "supervisor"


def assignment_root_dir(run_dir: Path, assignment: str) -> Path:
    return supervisor_root(run_dir) / "assignments" / assignment


def state_path(run_dir: Path) -> Path:
    return supervisor_root(run_dir) / "supervisor_state.json"


def status_path(run_dir: Path) -> Path:
    return supervisor_root(run_dir) / "status.md"


def lock_path(run_dir: Path) -> Path:
    return supervisor_root(run_dir) / "supervisor.lock"


@contextmanager
def supervisor_lock(run_dir: Path) -> Any:
    path = lock_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_cards(selected_assignments: Path) -> list[dict[str, Any]]:
    cards = read_jsonl(selected_assignments)
    if not cards:
        raise FileNotFoundError(f"no selected assignments found at {selected_assignments}")
    missing = [index for index, card in enumerate(cards, start=1) if not assignment_id(card)]
    if missing:
        raise ValueError(f"selected assignments missing assignment_id at rows: {missing[:10]}")
    return cards


def initial_status_for_card(run_dir: Path, card: dict[str, Any]) -> str:
    aid = assignment_id(card)
    final_judge = assignment_root_dir(run_dir, aid) / "judge" / "judge_candidates.jsonl"
    validation_accepted = assignment_root_dir(run_dir, aid) / "validation" / "validation_accepted_candidates.jsonl"
    output_path = Path(str(card.get("output_path") or ""))
    summary_path = Path(str(card.get("summary_path") or ""))
    if nonblank_line_count(final_judge):
        return "judged"
    if nonblank_line_count(validation_accepted):
        return "pending_judge"
    if nonblank_line_count(output_path):
        return "generated"
    return "pending_generation"


def load_or_init_state(args: argparse.Namespace, cards: list[dict[str, Any]]) -> dict[str, Any]:
    path = state_path(args.run_dir)
    card_ids = [assignment_id(card) for card in cards]
    if path.exists():
        state = read_json(path)
    else:
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "created_at": iso_now(),
            "run_dir": str(args.run_dir),
            "assignment_root": str(args.assignment_root),
            "selected_assignments": str(args.selected_assignments),
            "target_rows": args.target_rows,
            "max_active_agents": args.max_active_agents,
            "final_exported": False,
            "assignments": {},
        }
    assignments = state.setdefault("assignments", {})
    for ordinal, card in enumerate(cards, start=1):
        aid = assignment_id(card)
        item = assignments.setdefault(
            aid,
            {
                "assignment_id": aid,
                "ordinal": ordinal,
                "status": initial_status_for_card(args.run_dir, card),
                "generation_attempts": 0,
                "judge_attempts": 0,
                "generated_count": 0,
                "validation_ok_count": 0,
                "judge_accepted_count": 0,
                "judge_rejected_count": 0,
                "last_error": "",
            },
        )
        item.setdefault("ordinal", ordinal)
        if item.get("status") in {"running_generation", "running_judge"}:
            item["status"] = "generated" if nonblank_line_count(Path(str(card.get("output_path") or ""))) else "pending_generation"
            item["last_error"] = "stale running state reset on supervisor startup"
    state["assignment_order"] = card_ids
    state["updated_at"] = iso_now()
    return state


def build_validation_context(input_path: Path, run_dir: Path) -> dict[str, Any]:
    rows, _, _ = load_source_rows(input_path)
    clusters = {
        str(record.get("cluster_id")): record
        for record in read_jsonl(run_dir / "clusters.jsonl")
        if record.get("cluster_id")
    }
    return {
        "seen_chunk_ids": {chunk_id for row in rows for chunk_id in row.chunk_ids if chunk_id},
        "source_rows_by_key": source_rows_by_train_key(rows),
        "clusters": clusters,
        "exact_training_questions": {
            normalize_question(row.question) for row in rows if row.question
        },
    }


def collect_assignment(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    aid = assignment_id(card)
    out_dir = assignment_root_dir(args.run_dir, aid) / "collection"
    output_path = Path(str(card.get("output_path") or ""))
    summary_path = Path(str(card.get("summary_path") or ""))
    raw_rows = collect_load_jsonl(output_path)
    collected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    run_memory_records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(raw_rows, start=1):
        candidate, rejection = normalize_candidate(row, card=card, ordinal=ordinal)
        if candidate:
            collected.append(candidate)
            run_memory_records.append(run_memory_record(candidate, card=card, run_dir=args.run_dir))
        if rejection:
            rejected.append(rejection)

    write_collect_jsonl(collected, out_dir / "generated_candidates.jsonl")
    write_collect_jsonl(rejected, out_dir / "kimi_collection_rejected.jsonl")
    write_candidates_csv(collected, out_dir / "generated_candidates.csv")

    run_memory_written_count = 0
    if args.update_run_memory and run_memory_records:
        ledger = args.run_memory_ledger or args.assignment_root / "run_memory" / "generated_eval_rows.jsonl"
        existing = collect_load_jsonl(ledger)
        to_append = dedupe_run_memory_records(existing, run_memory_records)
        append_collect_jsonl(to_append, ledger)
        run_memory_written_count = len(to_append)

    summary = {
        "phase": "collect_kimi_assignment",
        "assignment_id": aid,
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "raw_row_count": len(raw_rows),
        "collected_count": len(collected),
        "rejected_count": len(rejected),
        "run_memory_written_count": run_memory_written_count,
        "rejection_reason_counts": dict(Counter(reason for row in rejected for reason in row.get("reasons", []))),
    }
    write_json(out_dir / "kimi_collection_summary.json", summary)
    return summary


def validate_assignment(card: dict[str, Any], args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    aid = assignment_id(card)
    in_path = assignment_root_dir(args.run_dir, aid) / "collection" / "generated_candidates.jsonl"
    out_dir = assignment_root_dir(args.run_dir, aid) / "validation"
    generated = read_jsonl(in_path)
    validated = [
        validate_one(
            record,
            seen_chunk_ids=context["seen_chunk_ids"],
            source_rows_by_key=context["source_rows_by_key"],
            clusters=context["clusters"],
            exact_training_questions=context["exact_training_questions"],
            similarity_threshold=args.similarity_threshold,
        )
        for record in generated
    ]
    accepted = [record for record in validated if record["validation"]["status"] == "ok"]
    rejected = [record for record in validated if record["validation"]["status"] != "ok"]
    write_jsonl(out_dir / "validated_candidates.jsonl", validated)
    write_jsonl(out_dir / "validation_accepted_candidates.jsonl", accepted)
    write_jsonl(out_dir / "validation_rejected_candidates.jsonl", rejected)
    summary = {
        "phase": "validate_assignment",
        "assignment_id": aid,
        "generated_count": len(generated),
        "validation_ok_count": len(accepted),
        "validation_reject_count": len(rejected),
        "rejection_reason_counts": dict(Counter(reason for record in rejected for reason in record["validation"]["reasons"])),
    }
    write_json(out_dir / "validation_summary.json", summary)
    return summary


def merge_judge_attempt(args: argparse.Namespace, aid: str, attempt_dir: Path) -> dict[str, Any]:
    final_dir = assignment_root_dir(args.run_dir, aid) / "judge"
    previous = {
        candidate_id(record): record
        for record in read_jsonl(final_dir / "judge_candidates.jsonl")
        if candidate_id(record)
    }
    for record in read_jsonl(attempt_dir / "judge_candidates.jsonl"):
        key = candidate_id(record)
        if key:
            previous[key] = record
    judged = sorted(previous.values(), key=candidate_id)
    accepted = [record for record in judged if (record.get("judge") or {}).get("status") == "accept"]
    rejected = [record for record in judged if (record.get("judge") or {}).get("status") != "accept"]
    errors = [
        {
            "candidate_id": candidate_id(record),
            "error": (record.get("judge") or {}).get("error", ""),
        }
        for record in rejected
        if (record.get("judge") or {}).get("reject_reason") == "judge_error"
    ]
    write_jsonl(final_dir / "judge_candidates.jsonl", judged)
    write_jsonl(final_dir / "judge_accepted_candidates.jsonl", accepted)
    write_jsonl(final_dir / "judge_rejected_candidates.jsonl", rejected)
    write_jsonl(final_dir / "judge_errors.jsonl", errors)
    summary = {
        "phase": "judge_assignment",
        "judge_version": JUDGE_VERSION,
        "assignment_id": aid,
        "input_candidate_count": len(judged),
        "judge_accepted_count": len(accepted),
        "judge_rejected_count": len(rejected),
        "error_count": len(errors),
        "reject_reason_counts": dict(Counter((record.get("judge") or {}).get("reject_reason") or "none" for record in rejected)),
    }
    write_json(final_dir / "judge_summary.json", summary)
    return summary


def rebuild_global_artifacts(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    order = state.get("assignment_order", [])
    generated: list[dict[str, Any]] = []
    collection_rejected: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    validation_accepted: list[dict[str, Any]] = []
    validation_rejected: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []
    judge_accepted: list[dict[str, Any]] = []
    judge_rejected: list[dict[str, Any]] = []
    judge_errors: list[dict[str, Any]] = []

    for aid in order:
        base = assignment_root_dir(args.run_dir, str(aid))
        generated.extend(read_jsonl(base / "collection" / "generated_candidates.jsonl"))
        collection_rejected.extend(read_jsonl(base / "collection" / "kimi_collection_rejected.jsonl"))
        validated.extend(read_jsonl(base / "validation" / "validated_candidates.jsonl"))
        validation_accepted.extend(read_jsonl(base / "validation" / "validation_accepted_candidates.jsonl"))
        validation_rejected.extend(read_jsonl(base / "validation" / "validation_rejected_candidates.jsonl"))
        judged.extend(read_jsonl(base / "judge" / "judge_candidates.jsonl"))
        judge_accepted.extend(read_jsonl(base / "judge" / "judge_accepted_candidates.jsonl"))
        judge_rejected.extend(read_jsonl(base / "judge" / "judge_rejected_candidates.jsonl"))
        judge_errors.extend(read_jsonl(base / "judge" / "judge_errors.jsonl"))

    write_jsonl(args.run_dir / "generated_candidates.jsonl", generated)
    write_candidates_csv(generated, args.run_dir / "generated_candidates.csv")
    write_jsonl(args.run_dir / "kimi_collection_rejected.jsonl", collection_rejected)
    write_jsonl(args.run_dir / "validated_candidates.jsonl", validated)
    write_jsonl(args.run_dir / "validation_accepted_candidates.jsonl", validation_accepted)
    write_jsonl(args.run_dir / "validation_rejected_candidates.jsonl", validation_rejected)
    write_jsonl(args.run_dir / "judge_candidates.jsonl", judged)
    write_jsonl(args.run_dir / "judge_accepted_candidates.jsonl", judge_accepted)
    write_jsonl(args.run_dir / "judge_rejected_candidates.jsonl", judge_rejected)
    write_jsonl(args.run_dir / "judge_errors.jsonl", judge_errors)

    generation_summary = {
        "phase": "generate",
        "source": "eval_supervisor_assignment_scoped",
        "selected_count": len(order),
        "generated_count": len(generated),
        "error_count": len(collection_rejected),
        "artifacts": {
            "generated_candidates": str(args.run_dir / "generated_candidates.jsonl"),
            "generated_candidates_csv": str(args.run_dir / "generated_candidates.csv"),
            "generation_errors": str(args.run_dir / "kimi_collection_rejected.jsonl"),
            "generation_summary": str(args.run_dir / "generation_summary.json"),
        },
        "sample_generated": generated[:3],
        "sample_errors": collection_rejected[:5],
    }
    collection_summary = {
        "phase": "collect_kimi",
        "source": "eval_supervisor_assignment_scoped",
        "assignment_count": len(order),
        "collected_count": len(generated),
        "rejected_count": len(collection_rejected),
        "rejection_reason_counts": dict(Counter(reason for row in collection_rejected for reason in row.get("reasons", []))),
    }
    validation_summary = {
        "phase": "validate",
        "generated_count": len(generated),
        "validation_ok_count": len(validation_accepted),
        "validation_reject_count": len(validation_rejected),
        "related_training_row_source": "candidate_seed_train_keys_union_cluster_seed_train_keys",
        "rejection_reason_counts": dict(Counter(reason for record in validation_rejected for reason in (record.get("validation") or {}).get("reasons", []))),
        "artifacts": {
            "validated_candidates": str(args.run_dir / "validated_candidates.jsonl"),
            "validation_accepted_candidates": str(args.run_dir / "validation_accepted_candidates.jsonl"),
            "validation_rejected_candidates": str(args.run_dir / "validation_rejected_candidates.jsonl"),
            "validation_summary": str(args.run_dir / "validation_summary.json"),
        },
    }
    judge_summary = {
        "phase": "judge",
        "judge_version": JUDGE_VERSION,
        "input_source": "assignment_scoped_validation_accepted_candidates",
        "model": args.llm_model,
        "input_candidate_count": len(validation_accepted),
        "judge_accepted_count": len(judge_accepted),
        "judge_rejected_count": len(judge_rejected),
        "error_count": len(judge_errors),
        "workers": "eval_supervisor_assignment_scoped",
        "reject_reason_counts": dict(Counter((record.get("judge") or {}).get("reject_reason") or "none" for record in judge_rejected)),
        "artifacts": {
            "judge_candidates": str(args.run_dir / "judge_candidates.jsonl"),
            "judge_accepted_candidates": str(args.run_dir / "judge_accepted_candidates.jsonl"),
            "judge_rejected_candidates": str(args.run_dir / "judge_rejected_candidates.jsonl"),
            "judge_errors": str(args.run_dir / "judge_errors.jsonl"),
            "judge_summary": str(args.run_dir / "judge_summary.json"),
        },
        "sample_accepted": judge_accepted[:3],
        "sample_rejected": judge_rejected[:5],
    }

    write_json(args.run_dir / "generation_summary.json", generation_summary)
    write_json(args.run_dir / "kimi_collection_summary.json", collection_summary)
    write_json(args.run_dir / "validation_summary.json", validation_summary)
    write_json(args.run_dir / "judge_summary.json", judge_summary)

    return {
        "generated_count": len(generated),
        "validation_ok_count": len(validation_accepted),
        "judge_accepted_count": len(judge_accepted),
        "judge_rejected_count": len(judge_rejected),
        "judge_error_count": len(judge_errors),
    }


def render_status(state: dict[str, Any], counts: dict[str, Any], running_generation: dict[str, Any], running_judges: dict[str, Any]) -> str:
    status_counts = Counter(item.get("status", "unknown") for item in state.get("assignments", {}).values())
    lines = [
        "# Eval-Question-Gen Supervisor Status",
        "",
        f"Updated: {iso_now()}",
        f"Run dir: `{state.get('run_dir', '')}`",
        f"Assignment root: `{state.get('assignment_root', '')}`",
        "",
        "## Counts",
        f"- target rows: {state.get('target_rows')}",
        f"- accepted rows: {counts.get('judge_accepted_count', 0)}",
        f"- generated rows: {counts.get('generated_count', 0)}",
        f"- validation ok rows: {counts.get('validation_ok_count', 0)}",
        f"- judge rejected rows: {counts.get('judge_rejected_count', 0)}",
        f"- judge error rows: {counts.get('judge_error_count', 0)}",
        "",
        "## Active Slots",
        f"- generation: {len(running_generation)}",
        f"- judge: {len(running_judges)}",
        f"- total: {len(running_generation) + len(running_judges)}/{state.get('max_active_agents')}",
        "",
        "## Assignment Status",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    if state.get("final_exported"):
        lines.extend(["", "## Final", "- export complete"])
    return "\n".join(lines).rstrip() + "\n"


def save_state(args: argparse.Namespace, state: dict[str, Any], counts: dict[str, Any], running_generation: dict[str, Any], running_judges: dict[str, Any]) -> None:
    state["updated_at"] = iso_now()
    state["counts"] = counts
    write_json(state_path(args.run_dir), state)
    status_path(args.run_dir).parent.mkdir(parents=True, exist_ok=True)
    status_path(args.run_dir).write_text(render_status(state, counts, running_generation, running_judges), encoding="utf-8")


def start_generation(card: dict[str, Any], args: argparse.Namespace, env: dict[str, str], state: dict[str, Any]) -> dict[str, Any]:
    aid = assignment_id(card)
    item = state["assignments"][aid]
    item["generation_attempts"] = int(item.get("generation_attempts") or 0) + 1
    attempt = item["generation_attempts"]
    log_dir = supervisor_root(args.run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{aid}.generation_attempt_{attempt:02d}.stdout.log"
    stderr_path = log_dir / f"{aid}.generation_attempt_{attempt:02d}.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    command = command_for_assignment(args.worker_command, card)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=stdout_handle,
        stderr=stderr_handle,
        env=env,
    )
    item.update(
        {
            "status": "running_generation",
            "generation_pid": process.pid,
            "generation_started_at": now_ts(),
            "last_error": "",
        }
    )
    append_event(supervisor_root(args.run_dir), "generation_started", assignment_id=aid, attempt=attempt, pid=process.pid)
    return {
        "process": process,
        "card": card,
        "started_at": time.time(),
        "stdout_handle": stdout_handle,
        "stderr_handle": stderr_handle,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "attempt": attempt,
    }


def close_generation(proc: dict[str, Any]) -> None:
    proc["stdout_handle"].close()
    proc["stderr_handle"].close()


def handle_generation_finished(
    card: dict[str, Any],
    args: argparse.Namespace,
    state: dict[str, Any],
    validation_context: dict[str, Any],
    *,
    returncode: int,
    reason: str,
) -> None:
    aid = assignment_id(card)
    item = state["assignments"][aid]
    collection = collect_assignment(card, args)
    validation = validate_assignment(card, args, validation_context)
    item["generated_count"] = collection["collected_count"]
    item["validation_ok_count"] = validation["validation_ok_count"]
    item["last_generation_returncode"] = returncode
    if validation["validation_ok_count"] > 0:
        item["status"] = "pending_judge"
        append_event(
            supervisor_root(args.run_dir),
            "generation_collected",
            assignment_id=aid,
            returncode=returncode,
            collected_count=collection["collected_count"],
            validation_ok_count=validation["validation_ok_count"],
        )
        return
    if returncode != 0 and int(item.get("generation_attempts") or 0) <= args.max_worker_retries:
        item["status"] = "pending_generation"
        item["last_error"] = reason
        append_event(supervisor_root(args.run_dir), "generation_retry_queued", assignment_id=aid, reason=reason)
        return
    item["status"] = "no_valid_rows"
    item["last_error"] = "no validation-accepted rows after generation"
    append_event(supervisor_root(args.run_dir), "generation_no_valid_rows", assignment_id=aid, reason=reason)


def judge_input_for_assignment(args: argparse.Namespace, aid: str, item: dict[str, Any]) -> Path:
    final_rejected = assignment_root_dir(args.run_dir, aid) / "judge" / "judge_rejected_candidates.jsonl"
    if int(item.get("judge_attempts") or 0) > 0 and final_rejected.exists():
        retry_records = [
            record
            for record in read_jsonl(final_rejected)
            if (record.get("judge") or {}).get("reject_reason") == "judge_error"
        ]
        if retry_records:
            retry_path = assignment_root_dir(args.run_dir, aid) / "judge" / f"judge_retry_input_{int(item.get('judge_attempts') or 0) + 1:02d}.jsonl"
            write_jsonl(retry_path, retry_records)
            return retry_path
    return assignment_root_dir(args.run_dir, aid) / "validation" / "validation_accepted_candidates.jsonl"


def start_judge(card: dict[str, Any], args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    aid = assignment_id(card)
    item = state["assignments"][aid]
    item["judge_attempts"] = int(item.get("judge_attempts") or 0) + 1
    attempt = item["judge_attempts"]
    attempt_dir = assignment_root_dir(args.run_dir, aid) / f"judge_attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    input_path = judge_input_for_assignment(args, aid, item)
    log_dir = supervisor_root(args.run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{aid}.judge_attempt_{attempt:02d}.stdout.log"
    stderr_path = log_dir / f"{aid}.judge_attempt_{attempt:02d}.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT_DIR / "judge_eval_rows.py"),
        "--run-dir",
        str(args.run_dir),
        "--assignment-root",
        str(args.assignment_root),
        "--selected-assignments",
        str(args.selected_assignments),
        "--input-candidates",
        str(input_path),
        "--output-dir",
        str(attempt_dir),
        "--assignment-id",
        aid,
        "--env-file",
        str(args.env_file),
        "--llm-model",
        args.llm_model,
        "--workers",
        "1",
        "--max-tokens",
        str(args.judge_max_tokens),
        "--llm-timeout-seconds",
        str(args.judge_llm_timeout_seconds),
        "--llm-retries",
        str(args.judge_llm_retries),
    ]
    if getattr(args, "judge_system_prompt", None) is not None:
        command.extend(["--system-prompt", str(args.judge_system_prompt)])
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=stdout_handle,
        stderr=stderr_handle,
        env=os.environ.copy(),
    )
    item.update(
        {
            "status": "running_judge",
            "judge_pid": process.pid,
            "judge_started_at": now_ts(),
            "last_error": "",
        }
    )
    append_event(supervisor_root(args.run_dir), "judge_started", assignment_id=aid, attempt=attempt, pid=process.pid, input_candidates=str(input_path))
    return {
        "process": process,
        "card": card,
        "attempt_dir": attempt_dir,
        "started_at": time.time(),
        "stdout_handle": stdout_handle,
        "stderr_handle": stderr_handle,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "attempt": attempt,
    }


def close_judge(proc: dict[str, Any]) -> None:
    proc["stdout_handle"].close()
    proc["stderr_handle"].close()


def handle_judge_finished(card: dict[str, Any], args: argparse.Namespace, state: dict[str, Any], *, attempt_dir: Path, returncode: int, reason: str) -> None:
    aid = assignment_id(card)
    item = state["assignments"][aid]
    if returncode != 0 and int(item.get("judge_attempts") or 0) <= args.max_judge_retries:
        item["status"] = "pending_judge"
        item["last_error"] = reason
        append_event(supervisor_root(args.run_dir), "judge_process_retry_queued", assignment_id=aid, reason=reason)
        return
    if returncode != 0:
        item["status"] = "judge_failed"
        item["last_error"] = reason
        append_event(supervisor_root(args.run_dir), "judge_process_failed", assignment_id=aid, reason=reason)
        return

    summary = merge_judge_attempt(args, aid, attempt_dir)
    item["judge_accepted_count"] = summary["judge_accepted_count"]
    item["judge_rejected_count"] = summary["judge_rejected_count"]
    if summary["error_count"] and int(item.get("judge_attempts") or 0) <= args.max_judge_retries:
        item["status"] = "pending_judge"
        item["last_error"] = f"{summary['error_count']} judge_error rows queued for retry"
        append_event(supervisor_root(args.run_dir), "judge_error_retry_queued", assignment_id=aid, error_count=summary["error_count"])
        return
    item["status"] = "judged"
    append_event(
        supervisor_root(args.run_dir),
        "judge_finished",
        assignment_id=aid,
        accepted=summary["judge_accepted_count"],
        rejected=summary["judge_rejected_count"],
        error_count=summary["error_count"],
    )


def poll_generation(
    running: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    state: dict[str, Any],
    validation_context: dict[str, Any],
) -> None:
    for aid, proc in list(running.items()):
        process = proc["process"]
        returncode = process.poll()
        if returncode is None and args.worker_timeout_seconds > 0:
            elapsed = time.time() - proc["started_at"]
            if elapsed > args.worker_timeout_seconds:
                process.terminate()
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
                close_generation(proc)
                running.pop(aid, None)
                handle_generation_finished(
                    proc["card"],
                    args,
                    state,
                    validation_context,
                    returncode=returncode,
                    reason=f"generation timed out after {int(args.worker_timeout_seconds)}s",
                )
                continue
        if returncode is None:
            continue
        close_generation(proc)
        running.pop(aid, None)
        reason = "generation completed" if returncode == 0 else f"generation exited {returncode}"
        handle_generation_finished(proc["card"], args, state, validation_context, returncode=returncode, reason=reason)


def poll_judges(running: dict[str, dict[str, Any]], args: argparse.Namespace, state: dict[str, Any]) -> None:
    for aid, proc in list(running.items()):
        process = proc["process"]
        returncode = process.poll()
        if returncode is None and args.judge_timeout_seconds > 0:
            elapsed = time.time() - proc["started_at"]
            if elapsed > args.judge_timeout_seconds:
                process.terminate()
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
                close_judge(proc)
                running.pop(aid, None)
                handle_judge_finished(
                    proc["card"],
                    args,
                    state,
                    attempt_dir=proc["attempt_dir"],
                    returncode=returncode,
                    reason=f"judge timed out after {int(args.judge_timeout_seconds)}s",
                )
                continue
        if returncode is None:
            continue
        close_judge(proc)
        running.pop(aid, None)
        reason = "judge completed" if returncode == 0 else f"judge exited {returncode}"
        handle_judge_finished(proc["card"], args, state, attempt_dir=proc["attempt_dir"], returncode=returncode, reason=reason)


def process_generated_backlog(
    cards_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    state: dict[str, Any],
    validation_context: dict[str, Any],
) -> None:
    for aid in state.get("assignment_order", []):
        item = state["assignments"][aid]
        if item.get("status") != "generated":
            continue
        handle_generation_finished(
            cards_by_id[aid],
            args,
            state,
            validation_context,
            returncode=0,
            reason="existing generated output collected on supervisor startup",
        )


def start_ready_work(
    cards_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    env: dict[str, str],
    state: dict[str, Any],
    counts: dict[str, Any],
    running_generation: dict[str, dict[str, Any]],
    running_judges: dict[str, dict[str, Any]],
) -> None:
    def active_slots() -> int:
        return len(running_generation) + len(running_judges)

    for aid in state.get("assignment_order", []):
        if active_slots() >= args.max_active_agents:
            return
        item = state["assignments"][aid]
        if item.get("status") != "pending_judge":
            continue
        running_judges[aid] = start_judge(cards_by_id[aid], args, state)

    if counts.get("judge_accepted_count", 0) >= args.target_rows and args.stop_after_target:
        return

    for aid in state.get("assignment_order", []):
        if active_slots() >= args.max_active_agents:
            return
        item = state["assignments"][aid]
        if item.get("status") != "pending_generation":
            continue
        running_generation[aid] = start_generation(cards_by_id[aid], args, env, state)


def unfinished_work_exists(state: dict[str, Any], running_generation: dict[str, Any], running_judges: dict[str, Any]) -> bool:
    if running_generation or running_judges:
        return True
    return any(
        item.get("status") in {"pending_generation", "running_generation", "generated", "pending_judge", "running_judge"}
        for item in state.get("assignments", {}).values()
    )


def judge_backlog_exists(state: dict[str, Any]) -> bool:
    return any(
        item.get("status") in {"generated", "pending_judge", "running_judge"}
        for item in state.get("assignments", {}).values()
    )


def should_export(args: argparse.Namespace, state: dict[str, Any], counts: dict[str, Any], running_generation: dict[str, Any], running_judges: dict[str, Any]) -> bool:
    if state.get("final_exported") or args.no_export:
        return False
    if running_generation or running_judges:
        return False
    if judge_backlog_exists(state):
        return False
    if counts.get("judge_accepted_count", 0) >= args.target_rows:
        return True
    return not unfinished_work_exists(state, running_generation, running_judges) and counts.get("judge_accepted_count", 0) > 0


def terminate_children(
    running: dict[str, dict[str, Any]],
    state: dict[str, Any],
    *,
    event_root: Path,
    event_name: str,
    fallback_status: str,
) -> None:
    for aid, proc in list(running.items()):
        process = proc["process"]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if "attempt_dir" in proc:
            close_judge(proc)
        else:
            close_generation(proc)
        item = state.get("assignments", {}).get(aid)
        if item:
            item["status"] = fallback_status
            item["last_error"] = "terminated because supervisor was interrupted"
        append_event(event_root, event_name, assignment_id=aid, pid=process.pid)
        running.pop(aid, None)


def run_supervisor(args: argparse.Namespace) -> int:
    if args.selected_assignments is None:
        args.selected_assignments = args.assignment_root / "selected_assignments.jsonl"
    cards = load_cards(args.selected_assignments)
    cards_by_id = {assignment_id(card): card for card in cards}
    validation_context = build_validation_context(args.input, args.run_dir)
    env = worker_env(args.env_file)
    state = load_or_init_state(args, cards)
    running_generation: dict[str, dict[str, Any]] = {}
    running_judges: dict[str, dict[str, Any]] = {}

    append_event(supervisor_root(args.run_dir), "supervisor_started", target_rows=args.target_rows, max_active_agents=args.max_active_agents)
    try:
        while True:
            poll_generation(running_generation, args, state, validation_context)
            poll_judges(running_judges, args, state)
            process_generated_backlog(cards_by_id, args, state, validation_context)
            counts = rebuild_global_artifacts(args, state)
            save_state(args, state, counts, running_generation, running_judges)

            if should_export(args, state, counts, running_generation, running_judges):
                summary = export_final_eval(run_dir=args.run_dir, eval_bank_path=args.eval_bank_path)
                state["final_exported"] = True
                state["export_summary"] = summary
                counts = rebuild_global_artifacts(args, state)
                save_state(args, state, counts, running_generation, running_judges)
                append_event(supervisor_root(args.run_dir), "export_complete", final_count=summary.get("final_count"))
                return 0

            start_ready_work(cards_by_id, args, env, state, counts, running_generation, running_judges)
            counts = rebuild_global_artifacts(args, state)
            save_state(args, state, counts, running_generation, running_judges)

            if not unfinished_work_exists(state, running_generation, running_judges):
                append_event(supervisor_root(args.run_dir), "supervisor_exhausted", accepted=counts.get("judge_accepted_count", 0))
                return 1 if counts.get("judge_accepted_count", 0) < args.target_rows else 0

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        root = supervisor_root(args.run_dir)
        terminate_children(
            running_generation,
            state,
            event_root=root,
            event_name="generation_terminated_on_interrupt",
            fallback_status="pending_generation",
        )
        terminate_children(
            running_judges,
            state,
            event_root=root,
            event_name="judge_terminated_on_interrupt",
            fallback_status="pending_judge",
        )
        counts = rebuild_global_artifacts(args, state)
        save_state(args, state, counts, running_generation, running_judges)
        append_event(root, "supervisor_interrupted")
        return 130


def print_status(args: argparse.Namespace) -> int:
    path = state_path(args.run_dir)
    if not path.exists():
        print(f"no supervisor state found: {path}")
        return 1
    state = read_json(path)
    status_md = status_path(args.run_dir)
    if status_md.exists():
        print(status_md.read_text(encoding="utf-8"))
    else:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--input", type=Path, default=DEFAULT_INPUT)
        target.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
        target.add_argument("--assignment-root", type=Path, default=DEFAULT_ASSIGNMENT_ROOT)
        target.add_argument("--selected-assignments", type=Path, default=None)
        target.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)

    run_parser = subparsers.add_parser("run", help="Run the assignment-scoped supervisor.")
    add_common(run_parser)
    run_parser.add_argument("--target-rows", type=int, default=200)
    run_parser.add_argument("--max-active-agents", type=int, default=3)
    run_parser.add_argument("--poll-seconds", type=float, default=10.0)
    run_parser.add_argument("--worker-timeout-seconds", type=float, default=1200.0)
    run_parser.add_argument("--judge-timeout-seconds", type=float, default=900.0)
    run_parser.add_argument("--max-worker-retries", type=int, default=1)
    run_parser.add_argument("--max-judge-retries", type=int, default=2)
    run_parser.add_argument("--worker-command", default=DEFAULT_WORKER_COMMAND)
    run_parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    run_parser.add_argument("--judge-max-tokens", type=int, default=2000)
    run_parser.add_argument("--judge-llm-timeout-seconds", type=int, default=180)
    run_parser.add_argument("--judge-llm-retries", type=int, default=2)
    run_parser.add_argument("--judge-system-prompt", type=Path, default=None, help="Path to a markdown file overriding the default judge system prompt.")
    run_parser.add_argument("--similarity-threshold", type=float, default=0.85)
    run_parser.add_argument("--eval-bank-path", type=Path, default=DEFAULT_EVAL_BANK)
    run_parser.add_argument("--run-memory-ledger", type=Path, default=None)
    run_parser.add_argument("--no-update-run-memory", dest="update_run_memory", action="store_false")
    run_parser.add_argument("--no-export", action="store_true")
    run_parser.add_argument("--run-all-assignments", dest="stop_after_target", action="store_false")
    run_parser.set_defaults(update_run_memory=True, stop_after_target=True)

    status_parser = subparsers.add_parser("status", help="Print the last supervisor status.")
    add_common(status_parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "status":
        return print_status(args)
    try:
        with supervisor_lock(args.run_dir):
            return run_supervisor(args)
    except BlockingIOError:
        print(f"another eval supervisor is already running for {args.run_dir}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
