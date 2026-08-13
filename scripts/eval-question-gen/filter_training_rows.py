#!/usr/bin/env python3
"""Filter training QA rows by quality, one batch per agent process.

Each process judges its batch of rows, appends the good rows to --output and the
rejected rows (with reasons) to --reject-log, then spawns the next process for the
next batch. Run with --start-row 0 to begin the chain.

Rows can be judged one at a time or in sub-batches (--rows-per-call) to reduce
API round trips on large datasets.

Example:
    python scripts/eval-question-gen/filter_training_rows.py \
        --input "questions/master_data - master_data.csv" \
        --output "questions/Filtered Master Questions.csv" \
        --reject-log "questions/Reject Master Questions.csv" \
        --start-row 0 \
        --batch-size 500 \
        --rows-per-call 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from llm_client import (
    LLMConfig,
    apply_env_file,
    call_llm_json,
    default_llm_api_key,
    default_llm_model,
    default_llm_url,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "FILTER_ROWS_PROMPT.md"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

BATCH_INSTRUCTION = """
You are about to receive MULTIPLE question-answer rows at once. Each row has a numeric ID.
Return a single JSON array where every element corresponds to one row ID, in the same order:

[
  {"row_id": 0, "decision": "KEEP", "reason": "..."},
  {"row_id": 1, "decision": "REJECT", "reason": "..."}
]

Do not skip any row_id. Be conservative: only reject obviously bad rows.
"""


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_user_prompt(rows: list[tuple[int, dict[str, str]]]) -> str:
    parts = []
    for local_id, row in rows:
        parts.append(
            f"[ROW {local_id}]\n"
            f"Question:\n{row.get('question', '')}\n\n"
            f"Answer:\n{row.get('answer', '')}\n"
        )
    parts.append("Return your decision as JSON:")
    return "\n---\n".join(parts)


def judge_batch(
    *,
    rows: list[tuple[int, dict[str, str]]],
    system_prompt: str,
    config: LLMConfig,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(rows)
    return call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        temperature=0.1,
        max_tokens=4000,
    )


def extract_json_from_text(text: str) -> Any:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = content.find(opener)
        end = content.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def parse_batch_decisions(result: dict[str, Any], local_ids: list[int]) -> dict[int, tuple[str, str]]:
    verdicts: dict[int, tuple[str, str]] = {}
    parsed = result.get("parsed") or extract_json_from_text(result.get("raw_content", ""))

    items: list[dict[str, Any]] = []
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "decisions" in parsed:
        items = parsed["decisions"]
    elif isinstance(parsed, dict):
        items = [parsed]

    for item in items:
        if not isinstance(item, dict):
            continue
        rid = item.get("row_id")
        if rid is None:
            rid = item.get("id")
        if rid is None:
            continue
        try:
            rid = int(rid)
        except (ValueError, TypeError):
            continue
        decision = str(item.get("decision") or item.get("verdict") or "REJECT").strip().upper()
        reason = str(item.get("reason") or "").strip()
        if decision not in {"KEEP", "REJECT"}:
            decision = "REJECT"
            reason = reason or "unparseable_decision"
        verdicts[rid] = (decision, reason)

    for rid in local_ids:
        if rid not in verdicts:
            verdicts[rid] = ("REJECT", "missing_from_batch_response")

    return verdicts


def build_next_command(
    *,
    script_path: Path,
    input_path: Path,
    output_path: Path,
    reject_path: Path | None,
    start_row: int,
    batch_size: int,
    rows_per_call: int,
    limit: int,
    model: str,
    env_file: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if reject_path:
        cmd += ["--reject-log", str(reject_path)]
    cmd += [
        "--start-row",
        str(start_row),
        "--batch-size",
        str(batch_size),
        "--rows-per-call",
        str(rows_per_call),
        "--limit",
        str(limit),
        "--model",
        model,
        "--env-file",
        env_file,
    ]
    return cmd


def resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    url = args.llm_url or default_llm_url()
    model = args.model or default_llm_model()
    api_key = args.llm_api_key if args.llm_api_key is not None else default_llm_api_key()
    if not url:
        raise SystemExit(
            "LLM URL is not configured. Set LLM_URL/LITELLM_BASE_URL in the env file or pass --llm-url."
        )
    return LLMConfig(
        url=url,
        model=model,
        api_key=api_key,
        timeout_seconds=args.llm_timeout_seconds,
        retries=args.llm_retries,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter training QA rows one batch at a time, self-spawning for the next batch."
    )
    parser.add_argument("--input", required=True, help="Source CSV path")
    parser.add_argument("--output", required=True, help="Output CSV for kept rows")
    parser.add_argument("--reject-log", help="Output CSV for rejected rows with reasons")
    parser.add_argument("--start-row", type=int, default=0, help="Zero-based row offset to start at")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per agent process")
    parser.add_argument(
        "--rows-per-call",
        type=int,
        default=5,
        help="Rows judged in a single LLM call (higher = fewer API calls)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max source rows to scan in total")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--llm-url", default=None, help="LLM chat-completions URL")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key")
    parser.add_argument("--llm-timeout-seconds", type=int, default=120, help="Per-call timeout")
    parser.add_argument("--llm-retries", type=int, default=2, help="Retries per call")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Env file to load LLM credentials from",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_PATH),
        help="Path to the filtering prompt markdown file",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Process one batch and exit without spawning the next agent",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without calling the LLM",
    )
    args = parser.parse_args(argv)

    apply_env_file(args.env_file)

    config = resolve_llm_config(args)
    base_prompt = load_prompt(Path(args.prompt_file))
    if args.rows_per_call > 1:
        system_prompt = BATCH_INSTRUCTION + "\n\n" + base_prompt
    else:
        system_prompt = base_prompt

    input_path = Path(args.input)
    output_path = Path(args.output)
    reject_path = Path(args.reject_log) if args.reject_log else None

    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    rows = load_csv(input_path)
    total_rows = len(rows)
    if total_rows == 0:
        print("[done] input CSV is empty")
        return 0

    limit = args.limit if args.limit is not None else total_rows
    start = max(0, args.start_row)
    end = min(start + args.batch_size, total_rows, limit)

    if start >= total_rows or start >= limit:
        print(f"[done] start-row {start} is beyond total/limit; nothing to do")
        return 0

    output_fieldnames = list(rows[0].keys())
    reject_fieldnames = output_fieldnames + ["_reject_reason", "_source_row_index"]

    print(
        f"[agent] pid={os.getpid()} start={start} end={end-1} "
        f"batch_size={end - start} rows_per_call={args.rows_per_call} "
        f"total={total_rows} limit={limit} model={config.model}",
        flush=True,
    )

    kept = 0
    rejected = 0
    errors = 0
    current_idx = start

    try:
        idx = start
        while idx < end:
            mini_batch_end = min(idx + args.rows_per_call, end)
            mini_batch = [(i - idx, rows[i]) for i in range(idx, mini_batch_end)]
            local_ids = [lid for lid, _ in mini_batch]
            source_indices = {lid: idx + offset for offset, (lid, _) in enumerate(mini_batch)}

            if args.dry_run:
                for lid, _ in mini_batch:
                    print(f"[dry-run] row {source_indices[lid] + 1}/{total_rows}: would judge", flush=True)
                idx = mini_batch_end
                continue

            verdicts: dict[int, tuple[str, str]] = {}
            try:
                result = judge_batch(
                    rows=mini_batch,
                    system_prompt=system_prompt,
                    config=config,
                )
                verdicts = parse_batch_decisions(result, local_ids)
            except Exception:
                for lid, row in mini_batch:
                    source_idx = source_indices[lid]
                    try:
                        single_result = judge_batch(
                            rows=[(0, row)],
                            system_prompt=system_prompt,
                            config=config,
                        )
                        single_verdicts = parse_batch_decisions(single_result, [0])
                        verdicts[lid] = single_verdicts.get(0, ("REJECT", "missing_single_verdict"))
                    except Exception as single_exc:
                        verdicts[lid] = ("REJECT", f"llm_error: {single_exc}")
                        errors += 1

            for lid, row in mini_batch:
                decision, reason = verdicts.get(lid, ("REJECT", "missing_verdict"))
                source_idx = source_indices[lid]
                if decision == "KEEP":
                    kept += 1
                    append_csv_row(output_path, row, output_fieldnames)
                else:
                    rejected += 1
                    if reject_path:
                        reject_row = {
                            **row,
                            "_reject_reason": reason,
                            "_source_row_index": source_idx,
                        }
                        append_csv_row(reject_path, reject_row, reject_fieldnames)

            idx = mini_batch_end
            current_idx = idx
            processed_in_batch = idx - start
            if processed_in_batch % 50 == 0 or idx == end:
                print(
                    f"[progress] row {idx}/{total_rows} "
                    f"batch={processed_in_batch}/{end - start} "
                    f"kept={kept} rejected={rejected} errors={errors}",
                    flush=True,
                )
                time.sleep(0)
    except KeyboardInterrupt:
        print(f"\n[interrupted] agent exiting at row {current_idx + 1}", flush=True)
        return 130

    print(
        f"[done] agent pid={os.getpid()} rows {start}-{end - 1}: "
        f"kept={kept} rejected={rejected} errors={errors}",
        flush=True,
    )

    if not args.no_spawn and end < min(total_rows, limit):
        next_cmd = build_next_command(
            script_path=Path(__file__).resolve(),
            input_path=input_path,
            output_path=output_path,
            reject_path=reject_path,
            start_row=end,
            batch_size=args.batch_size,
            rows_per_call=args.rows_per_call,
            limit=limit,
            model=config.model,
            env_file=args.env_file,
        )
        print(f"[spawn] {' '.join(next_cmd)}", flush=True)
        subprocess.Popen(next_cmd)
    else:
        print("[done] no more batches to spawn", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
