#!/usr/bin/env python3
"""Fast cluster-local LLM judge for Eval-Question-Gen rows."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_paths import DEFAULT_ASSIGNMENT_ROOT, DEFAULT_ENV_FILE, DEFAULT_RUN_DIR
from llm_client import (
    LLMConfig,
    apply_env_file,
    call_llm_json,
    default_llm_api_key,
    default_llm_model,
    default_llm_url,
)


JUDGE_VERSION = "cluster_local_v2_run_dedupe"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}
DEFAULT_MIN_DIFFICULTY = "medium"

MAX_JUDGE_ROW_ATTEMPTS = 3
MAX_JUDGE_TOKENS_CEILING = 8000

SYSTEM_PROMPT = """You are a fast, strict judge for Eval-Question-Gen.

Evaluate one generated eval row using only the provided cited chunks.
The related training rows and same-run previous questions are for distinctness
checks only; they are not evidence for the answer.

Return only JSON:
{
  "verdict": "accept|reject",
  "answer_support": "supported|unsupported",
  "distinctness": "distinct|too_similar",
  "citation_quality": "sufficient|insufficient|over_cited",
  "eval_quality": "useful|weak|unfair",
  "difficulty": "easy|medium|hard",
  "supporting_chunk_ids": ["chunk-id"],
  "too_similar_refs": [],
  "reason": "one short reason"
}

Difficulty scoring:
- easy: the question asks for a plainly stated fact, date, name, amount,
  section number, case citation, or single-sentence extraction.
- medium: the question requires connecting two facts, applying a stated rule
  or condition, comparing two positions, or explaining a stated rationale.
- hard: the question requires synthesizing multiple parts of the chunks,
  tracing reasoning, resolving an apparent tension, or drawing a non-obvious
  inference that is explicitly grounded in the text.

Before choosing the verdict, internally check the answer's key factual/legal
claims against the cited chunks. Do not output a claim-by-claim proof. Keep the
JSON compact and do not wrap it in markdown fences.

Accept only if the answer is fully supported by the cited chunks, the cited
chunks are sufficient without obvious over-citation, the question is
meaningfully different from related training rows and same-run previous
questions, the row is useful as an eval item, and the difficulty is at least
medium. Reject cosmetic rewrites, unsupported answers, questions needing unseen
context, unfair traps, boilerplate rows, and easy single-fact lookups.
"""


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


def env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    url = args.llm_url or default_llm_url()
    model = args.llm_model or default_llm_model()
    api_key = args.llm_api_key if args.llm_api_key is not None else default_llm_api_key()
    if not url:
        raise SystemExit(
            "LLM URL is not configured. Set LLM_URL/LITELLM_BASE_URL in the env "
            "file or pass --llm-url."
        )
    timeout_seconds = args.llm_timeout_seconds or env_int(
        ("JUDGE_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS", "LITELLM_TIMEOUT_SECONDS"),
        120,
    )
    retries = args.llm_retries if args.llm_retries is not None else env_int(
        ("JUDGE_LLM_RETRIES", "LLM_RETRIES", "LITELLM_RETRIES"),
        1,
    )
    return LLMConfig(
        url=url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )


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


def truncate(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value or "") if len(token) > 1}


def jaccard(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def training_row_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    value = str(row.get("train_key") or "")
    return (0 if value else 1, value)


def load_cards(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(card.get("assignment_id")): card
        for card in load_jsonl(path)
        if card.get("assignment_id")
    }


def load_chunks_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("chunk_id")): record
        for record in load_jsonl(path)
        if record.get("chunk_id") and record.get("hydration_status") == "ok"
    }


def compact_training_rows(
    rows: list[dict[str, Any]],
    *,
    question: str,
    cited_chunk_ids: list[str],
    max_rows: int,
) -> list[dict[str, Any]]:
    cited = set(cited_chunk_ids)

    def relevance(row: dict[str, Any]) -> tuple[int, float, tuple[int, str]]:
        row_chunks = set(normalize_string_list(row.get("chunk_ids")))
        overlap = len(cited & row_chunks)
        similarity = jaccard(question, str(row.get("question") or ""))
        return (-overlap, -similarity, training_row_sort_key(row))

    compact: list[dict[str, Any]] = []
    for row in sorted(rows, key=relevance)[:max_rows]:
        row_chunks = normalize_string_list(row.get("chunk_ids"))
        compact.append(
            {
                "train_key": row.get("train_key"),
                "question": truncate(row.get("question"), 500),
                "answer": truncate(row.get("answer"), 500),
                "chunk_ids": row_chunks,
                "overlap_chunk_ids": sorted(cited & set(row_chunks)),
            }
        )
    return compact


def compact_run_previous_questions(
    rows: list[dict[str, Any]],
    *,
    question: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: -jaccard(question, str(row.get("question") or "")),
    )
    return [
        {
            "question": truncate(row.get("question"), 400),
            "answer_short": truncate(row.get("answer_short") or row.get("answer"), 300),
            "used_chunk_ids": normalize_string_list(row.get("used_chunk_ids")),
            "assignment_id": row.get("assignment_id", ""),
            "status": row.get("status", ""),
        }
        for row in rows[:max_rows]
        if str(row.get("question") or "").strip()
    ]


def reject_record(record: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    return {
        **record,
        "judge": {
            "status": "reject",
            "judge_version": JUDGE_VERSION,
            "reject_reason": reason,
            "reasons": [reason],
            **extra,
        },
    }


def build_packet(
    record: dict[str, Any],
    *,
    cards: dict[str, dict[str, Any]],
    max_chunk_chars: int,
    max_training_rows: int,
    max_run_previous_questions: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    internal = record.get("internal") if isinstance(record.get("internal"), dict) else {}
    assignment_id = str(internal.get("assignment_id") or "").strip()
    if not assignment_id:
        return None, reject_record(record, "missing_assignment_id")
    card = cards.get(assignment_id)
    if not card:
        return None, reject_record(record, "assignment_not_found")

    allowed_chunk_ids = set(normalize_string_list(card.get("allowed_chunk_ids")))
    cited_chunk_ids = normalize_string_list(record.get("chunk_ids"))
    if not cited_chunk_ids:
        return None, reject_record(record, "empty_cited_chunks")
    outside_allowed = sorted(set(cited_chunk_ids) - allowed_chunk_ids)
    if outside_allowed:
        return None, reject_record(
            record,
            "cited_chunk_outside_assignment",
            outside_allowed_chunk_ids=outside_allowed,
        )

    artifacts = card.get("artifact_paths") if isinstance(card.get("artifact_paths"), dict) else {}
    evidence_path = Path(str(artifacts.get("assignment_hydrated_chunks_jsonl") or ""))
    if not evidence_path.exists():
        return None, reject_record(record, "missing_assignment_evidence_bundle", evidence_path=str(evidence_path))
    chunks_by_id = load_chunks_by_id(evidence_path)
    missing_cited_chunks = [chunk_id for chunk_id in cited_chunk_ids if chunk_id not in chunks_by_id]
    if missing_cited_chunks:
        return None, reject_record(
            record,
            "missing_cited_chunk_text",
            missing_cited_chunk_ids=missing_cited_chunks,
            evidence_path=str(evidence_path),
        )

    training_path_value = str(artifacts.get("related_training_rows_jsonl") or "").strip()
    if not training_path_value:
        return None, reject_record(record, "missing_related_training_rows_artifact")
    training_path = Path(training_path_value)
    if not training_path.exists():
        return None, reject_record(
            record,
            "missing_related_training_rows_bundle",
            training_path=str(training_path),
        )
    related_training_rows = load_jsonl(training_path)
    if not related_training_rows:
        return None, reject_record(
            record,
            "empty_related_training_rows_bundle",
            training_path=str(training_path),
        )

    question = str(record.get("question") or "")
    compact_training = compact_training_rows(
        related_training_rows,
        question=question,
        cited_chunk_ids=cited_chunk_ids,
        max_rows=max_training_rows,
    )
    run_previous_questions = card.get("run_previous_questions_to_avoid")
    if not isinstance(run_previous_questions, list):
        run_previous_questions = []
    packet = {
        "candidate": {
            "question": str(record.get("question") or "").strip(),
            "answer": str(record.get("answer") or "").strip(),
            "docIds": normalize_string_list(record.get("docIds")),
            "chunk_ids": cited_chunk_ids,
            "eval_question_hash": str(record.get("eval_question_hash") or "").strip(),
            "doc_key": str(record.get("doc_key") or "").strip(),
            "seed_train_keys": normalize_string_list(record.get("seed_train_keys")),
        },
        "assignment": {
            "assignment_id": assignment_id,
            "cluster_id": internal.get("cluster_id"),
            "microcluster_id": internal.get("microcluster_id"),
            "cluster_kind": internal.get("cluster_kind"),
            "allowed_chunk_ids": sorted(allowed_chunk_ids),
        },
        "cited_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id"),
                "chunk_index": chunk.get("chunk_index"),
                "pages": chunk.get("pages", []),
                "labels": chunk.get("labels", []),
                "doc": chunk.get("doc", {}),
                "text": truncate(chunk.get("text"), max_chunk_chars),
            }
            for chunk_id in cited_chunk_ids
            for chunk in [chunks_by_id[chunk_id]]
        ],
        "related_training_rows": {
            "artifact_path": str(training_path),
            "total_related_row_count": len(related_training_rows),
            "shown_for_distinctness_count": len(compact_training),
            "selection_rule": (
                "highest cited-chunk overlap, then highest token similarity to "
                "candidate question, then train key"
            ),
            "shown_for_distinctness": compact_training,
        },
        "run_previous_questions_to_avoid": compact_run_previous_questions(
            run_previous_questions,
            question=question,
            max_rows=max_run_previous_questions,
        ),
        "judge_rules": {
            "evidence_boundary": "Use only cited_chunks as evidence for answer support.",
            "training_rows_are_evidence": False,
            "run_previous_questions_are_evidence": False,
            "reject_on_unseen_context": True,
            "reject_on_cosmetic_training_rewrite": True,
        },
    }
    return packet, None


def normalize_judge(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "verdict": "reject",
            "answer_support": "unsupported",
            "distinctness": "too_similar",
            "citation_quality": "insufficient",
            "eval_quality": "weak",
            "supporting_chunk_ids": [],
            "too_similar_refs": [],
            "reason": "judge_response_not_object",
        }
    raw_reasons = parsed.get("reasons", [])
    if not isinstance(raw_reasons, list):
        raw_reasons = [str(raw_reasons)]
    reason = str(parsed.get("reason") or "").strip()
    if not reason and raw_reasons:
        reason = str(raw_reasons[0]).strip()

    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in {"accept", "reject"}:
        verdict = "reject"
        reason = reason or "invalid_verdict"

    answer_support = str(
        parsed.get("answer_support") or parsed.get("grounding") or ""
    ).strip().lower()
    if answer_support == "partial":
        answer_support = "unsupported"

    citation_quality = str(parsed.get("citation_quality") or "").strip().lower()
    if not citation_quality:
        chunk_sufficiency = str(parsed.get("chunk_sufficiency") or "").strip().lower()
        citation_quality = "sufficient" if chunk_sufficiency == "sufficient" else "insufficient"
    if normalize_string_list(parsed.get("unnecessary_chunk_ids")):
        citation_quality = "over_cited"

    eval_quality = str(parsed.get("eval_quality") or "").strip().lower()
    if not eval_quality:
        usefulness = str(parsed.get("usefulness") or "").strip().lower()
        trap_fairness = str(parsed.get("trap_fairness") or "").strip().lower()
        if trap_fairness == "unfair":
            eval_quality = "unfair"
        else:
            eval_quality = "useful" if usefulness == "useful" else "weak"

    difficulty = str(parsed.get("difficulty") or "").strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "easy"

    too_similar_refs = normalize_string_list(parsed.get("too_similar_refs"))
    old_distinctness = parsed.get("distinctness_check", {})
    if isinstance(old_distinctness, dict):
        too_similar_refs.extend(
            normalize_string_list(old_distinctness.get("too_similar_run_previous_assignment_ids"))
        )

    return {
        "verdict": verdict,
        "answer_support": answer_support,
        "distinctness": str(parsed.get("distinctness") or "").strip().lower(),
        "citation_quality": citation_quality,
        "eval_quality": eval_quality,
        "difficulty": difficulty,
        "supporting_chunk_ids": normalize_string_list(parsed.get("supporting_chunk_ids")),
        "too_similar_refs": sorted(set(too_similar_refs)),
        "reason": truncate(reason, 500),
    }


def judge_gate_reasons(
    judge: dict[str, Any],
    *,
    cited_chunk_ids: list[str],
    min_difficulty: str = DEFAULT_MIN_DIFFICULTY,
) -> list[str]:
    reasons: list[str] = []
    if judge["verdict"] != "accept":
        reasons.append("judge_verdict_reject")
    if judge["answer_support"] != "supported":
        reasons.append("answer_not_supported")
    if judge["distinctness"] != "distinct":
        reasons.append("too_similar")
    if judge["citation_quality"] != "sufficient":
        reasons.append(f"citation_{judge['citation_quality'] or 'invalid'}")
    if judge["eval_quality"] != "useful":
        reasons.append(f"eval_quality_{judge['eval_quality'] or 'invalid'}")
    minimum_rank = DIFFICULTY_RANK.get(min_difficulty, DIFFICULTY_RANK[DEFAULT_MIN_DIFFICULTY])
    candidate_rank = DIFFICULTY_RANK.get(str(judge.get("difficulty") or ""), -1)
    if candidate_rank < minimum_rank:
        reasons.append("difficulty_too_low")

    cited = set(cited_chunk_ids)
    supporting = set(normalize_string_list(judge.get("supporting_chunk_ids")))
    if not supporting:
        reasons.append("missing_supporting_chunk_ids")
    elif not supporting.issubset(cited):
        reasons.append("supporting_chunk_outside_cited")
    return reasons


def judge_passes(
    judge: dict[str, Any],
    *,
    cited_chunk_ids: list[str],
    min_difficulty: str = DEFAULT_MIN_DIFFICULTY,
) -> bool:
    return not judge_gate_reasons(
        judge,
        cited_chunk_ids=cited_chunk_ids,
        min_difficulty=min_difficulty,
    )


def judge_one(
    record: dict[str, Any],
    *,
    cards: dict[str, dict[str, Any]],
    llm_config: LLMConfig,
    max_chunk_chars: int,
    max_training_rows: int,
    max_run_previous_questions: int,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    min_difficulty: str,
) -> dict[str, Any]:
    packet, preflight_reject = build_packet(
        record,
        cards=cards,
        max_chunk_chars=max_chunk_chars,
        max_training_rows=max_training_rows,
        max_run_previous_questions=max_run_previous_questions,
    )
    if preflight_reject:
        return preflight_reject
    assert packet is not None
    current_max_tokens = max_tokens
    last_error: Exception | None = None
    for row_attempt in range(1, MAX_JUDGE_ROW_ATTEMPTS + 1):
        started_at = time.time()
        try:
            result = call_llm_json(
                system_prompt=system_prompt,
                user_prompt="Judge this row quickly and strictly:\n\n"
                + json.dumps(packet, indent=2, ensure_ascii=False),
                config=llm_config,
                temperature=temperature,
                max_tokens=current_max_tokens,
            )
            judge = normalize_judge(result.get("parsed"))
            cited_chunk_ids = normalize_string_list(record.get("chunk_ids"))
            gate_reasons = judge_gate_reasons(
                judge,
                cited_chunk_ids=cited_chunk_ids,
                min_difficulty=min_difficulty,
            )
            status = "accept" if not gate_reasons else "reject"
            reject_reason = "" if status == "accept" else gate_reasons[0]
            return {
                **record,
                "judge": {
                    "status": status,
                    "judge_version": JUDGE_VERSION,
                    "model": llm_config.model,
                    "duration_ms": result.get("duration_ms"),
                    "total_duration_ms": round((time.time() - started_at) * 1000),
                    "attempt": result.get("attempt"),
                    "row_attempts": row_attempt,
                    "reject_reason": reject_reason,
                    "gate_reasons": gate_reasons,
                    **judge,
                },
            }
        except Exception as exc:
            last_error = exc
            if row_attempt < MAX_JUDGE_ROW_ATTEMPTS:
                current_max_tokens = min(current_max_tokens * 2, MAX_JUDGE_TOKENS_CEILING)
                time.sleep(0.5)
                continue
            break

    return reject_record(
        record,
        "judge_error",
        error=f"{type(last_error).__name__}: {last_error}",
        row_attempts=MAX_JUDGE_ROW_ATTEMPTS,
    )


def run_judge(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file:
        apply_env_file(str(args.env_file))
    if args.selected_assignments is None:
        args.selected_assignments = args.assignment_root / "selected_assignments.jsonl"
    cards = load_cards(args.selected_assignments)
    input_candidates_path = args.input_candidates or args.run_dir / "validation_accepted_candidates.jsonl"
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(input_candidates_path)
    if args.assignment_id:
        wanted = set(args.assignment_id)
        candidates = [
            record
            for record in candidates
            if str((record.get("internal") or {}).get("assignment_id") or "") in wanted
        ]
    if args.limit_candidates is not None:
        candidates = candidates[: max(0, args.limit_candidates)]

    if args.dry_run:
        packets = []
        preflight_rejections = []
        for record in candidates[: max(0, args.sample_packets)]:
            packet, rejection = build_packet(
                record,
                cards=cards,
                max_chunk_chars=args.max_chars_per_chunk,
                max_training_rows=args.max_training_rows,
                max_run_previous_questions=args.max_run_previous_questions,
            )
            if packet:
                packets.append(packet)
            if rejection:
                preflight_rejections.append(rejection)
        return {
            "phase": "judge",
            "judge_version": JUDGE_VERSION,
            "dry_run": True,
            "input_candidate_count": len(candidates),
            "selected_assignment_count": len(cards),
            "sample_packet_count": len(packets),
            "preflight_rejection_count": len(preflight_rejections),
            "min_difficulty": args.min_difficulty,
            "sample_packets": packets,
            "sample_preflight_rejections": preflight_rejections[:5],
        }

    llm_config = resolve_llm_config(args)
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt is not None:
        system_prompt = args.system_prompt.read_text(encoding="utf-8")
    judged: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if args.workers <= 1:
        for record in candidates:
            try:
                judged.append(
                    judge_one(
                        record,
                        cards=cards,
                        llm_config=llm_config,
                        max_chunk_chars=args.max_chars_per_chunk,
                        max_training_rows=args.max_training_rows,
                        max_run_previous_questions=args.max_run_previous_questions,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        system_prompt=system_prompt,
                        min_difficulty=args.min_difficulty,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "candidate_id": (record.get("internal") or {}).get("candidate_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                judged.append(reject_record(record, "judge_error", error=str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_record = {
                executor.submit(
                    judge_one,
                    record,
                    cards=cards,
                    llm_config=llm_config,
                    max_chunk_chars=args.max_chars_per_chunk,
                    max_training_rows=args.max_training_rows,
                    max_run_previous_questions=args.max_run_previous_questions,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    system_prompt=system_prompt,
                    min_difficulty=args.min_difficulty,
                ): record
                for record in candidates
            }
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    judged.append(future.result())
                except Exception as exc:
                    errors.append(
                        {
                            "candidate_id": (record.get("internal") or {}).get("candidate_id"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    judged.append(reject_record(record, "judge_error", error=str(exc)))

    judged.sort(key=lambda record: str((record.get("internal") or {}).get("candidate_id") or ""))
    accepted = [record for record in judged if (record.get("judge") or {}).get("status") == "accept"]
    rejected = [record for record in judged if (record.get("judge") or {}).get("status") != "accept"]

    judged_path = output_dir / "judge_candidates.jsonl"
    accepted_path = output_dir / "judge_accepted_candidates.jsonl"
    rejected_path = output_dir / "judge_rejected_candidates.jsonl"
    errors_path = output_dir / "judge_errors.jsonl"
    summary_path = output_dir / "judge_summary.json"
    write_jsonl(judged, judged_path)
    write_jsonl(accepted, accepted_path)
    write_jsonl(rejected, rejected_path)
    write_jsonl(errors, errors_path)

    summary = {
        "phase": "judge",
        "judge_version": JUDGE_VERSION,
        "input_source": str(input_candidates_path),
        "model": llm_config.model,
        "input_candidate_count": len(candidates),
        "judge_accepted_count": len(accepted),
        "judge_rejected_count": len(rejected),
        "error_count": len(errors),
        "workers": args.workers,
        "limits": {
            "max_chars_per_chunk": args.max_chars_per_chunk,
            "max_training_rows": args.max_training_rows,
            "max_run_previous_questions": args.max_run_previous_questions,
            "max_tokens": args.max_tokens,
            "timeout_seconds": llm_config.timeout_seconds,
            "retries": llm_config.retries,
            "min_difficulty": args.min_difficulty,
        },
        "reject_reason_counts": dict(
            Counter((record.get("judge") or {}).get("reject_reason") or "none" for record in rejected)
        ),
        "artifacts": {
            "judge_candidates": str(judged_path),
            "judge_accepted_candidates": str(accepted_path),
            "judge_rejected_candidates": str(rejected_path),
            "judge_errors": str(errors_path),
            "judge_summary": str(summary_path),
        },
        "sample_accepted": accepted[:3],
        "sample_rejected": rejected[:5],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--assignment-root", type=Path, default=DEFAULT_ASSIGNMENT_ROOT)
    parser.add_argument("--selected-assignments", type=Path, default=None)
    parser.add_argument("--input-candidates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--assignment-id", action="append", default=[])
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--llm-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout-seconds", type=int, default=None)
    parser.add_argument("--llm-retries", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1600)
    parser.add_argument("--max-training-rows", type=int, default=10)
    parser.add_argument("--max-run-previous-questions", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--min-difficulty", choices=sorted(DIFFICULTY_RANK), default=DEFAULT_MIN_DIFFICULTY)
    parser.add_argument("--system-prompt", type=Path, default=None, help="Path to a markdown file overriding the default judge system prompt.")
    parser.add_argument("--sample-packets", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        summary = run_judge(parse_args())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
