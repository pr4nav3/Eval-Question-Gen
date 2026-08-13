#!/usr/bin/env python3
"""Run prepared Kimi seen-chunk eval assignments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pipeline_paths import DEFAULT_ASSIGNMENT_ROOT, DEFAULT_ENV_FILE, REPO_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = DEFAULT_ASSIGNMENT_ROOT
DEFAULT_WORKER_COMMAND = 'opencode run -m litellm/private-large "Read and execute the run prompt at {prompt_path}"'


class FormatValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"unknown worker command placeholder `{key}`")


def now_batch_id() -> str:
    return time.strftime("kimi_eval_batch_%Y%m%d_%H%M%S")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_ws(value: Any) -> str:
    return str(value or "").strip()


def nonblank_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_env_file(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[key] = value
    return loaded


def worker_env(env_file: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(env_file))
    api_key = (
        env.get("LITELLM_API_KEY")
        or env.get("JUSPAY_API_KEY")
        or env.get("OPENAI_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            "No API key found after loading env file. Expected LITELLM_API_KEY in .env."
        )
    env["LITELLM_API_KEY"] = api_key
    env["JUSPAY_API_KEY"] = api_key
    env["OPENAI_API_KEY"] = api_key
    return env


def command_for_assignment(template: str, card: dict[str, Any]) -> list[str]:
    values = FormatValues(
        {
            "assignment_id": normalize_ws(card.get("assignment_id")),
            "assignment_path": normalize_ws(card.get("assignment_path")),
            "prompt_path": normalize_ws(card.get("prompt_path")),
            "output_path": normalize_ws(card.get("output_path")),
            "summary_path": normalize_ws(card.get("summary_path")),
            "agent_instructions": normalize_ws(card.get("agent_instructions")),
            "cwd": str(REPO_ROOT),
        }
    )
    try:
        rendered = template.format_map(values)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    return shlex.split(rendered)


def selected_cards(args: argparse.Namespace) -> list[dict[str, Any]]:
    cards = load_jsonl(args.selected_assignments)
    if not cards:
        raise FileNotFoundError(f"no selected assignments found at {args.selected_assignments}")
    wanted = set(args.assignment_id)
    if wanted:
        cards = [card for card in cards if normalize_ws(card.get("assignment_id")) in wanted]
        missing = sorted(wanted.difference(normalize_ws(card.get("assignment_id")) for card in cards))
        if missing:
            raise ValueError(f"assignment(s) not found: {', '.join(missing)}")
    if args.skip_completed:
        cards = [
            card
            for card in cards
            if nonblank_line_count(Path(normalize_ws(card.get("output_path")))) == 0
            or not Path(normalize_ws(card.get("summary_path"))).exists()
        ]
    if args.max_assignments > 0:
        cards = cards[: args.max_assignments]
    return cards


def run_cards(args: argparse.Namespace, cards: list[dict[str, Any]]) -> dict[str, Any]:
    batch_dir = args.output_root / "batch_runs" / args.batch_id
    event_log = batch_dir / "events.jsonl"
    batch_dir.mkdir(parents=True, exist_ok=True)
    env = worker_env(args.env_file)

    pending = list(cards)
    running: list[tuple[subprocess.Popen[str], dict[str, Any], Any, Any, float]] = []
    started = 0
    finished = 0
    timed_out = 0
    failed = 0
    missing_summary = 0

    if args.dry_run:
        planned = []
        for card in cards:
            command = command_for_assignment(args.worker_command, card)
            planned.append(
                {
                    "assignment_id": normalize_ws(card.get("assignment_id")),
                    "command": command,
                    "prompt_path": normalize_ws(card.get("prompt_path")),
                }
            )
        summary = {
            "batch_id": args.batch_id,
            "dry_run": True,
            "planned_count": len(planned),
            "parallel": args.parallel,
            "worker_timeout_seconds": args.worker_timeout_seconds,
            "sample_planned": planned[:5],
        }
        (batch_dir / "batch_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary

    while pending or running:
        while pending and len(running) < args.parallel:
            card = pending.pop(0)
            assignment_id = normalize_ws(card.get("assignment_id"))
            command = command_for_assignment(args.worker_command, card)
            stdout_path = batch_dir / f"{assignment_id}.stdout.log"
            stderr_path = batch_dir / f"{assignment_id}.stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
            )
            running.append((process, card, stdout_handle, stderr_handle, time.time()))
            started += 1
            append_jsonl(
                event_log,
                {
                    "ts": int(time.time()),
                    "event": "worker_started",
                    "assignment_id": assignment_id,
                    "pid": process.pid,
                    "prompt_path": normalize_ws(card.get("prompt_path")),
                    "output_path": normalize_ws(card.get("output_path")),
                    "summary_path": normalize_ws(card.get("summary_path")),
                },
            )

        time.sleep(args.poll_seconds)
        still_running: list[tuple[subprocess.Popen[str], dict[str, Any], Any, Any, float]] = []
        for process, card, stdout_handle, stderr_handle, started_at in running:
            assignment_id = normalize_ws(card.get("assignment_id"))
            returncode = process.poll()
            if returncode is None and args.worker_timeout_seconds > 0:
                elapsed = time.time() - started_at
                if elapsed > args.worker_timeout_seconds:
                    process.terminate()
                    try:
                        returncode = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        returncode = process.wait()
                    stdout_handle.close()
                    stderr_handle.close()
                    timed_out += 1
                    failed += 1
                    append_jsonl(
                        event_log,
                        {
                            "ts": int(time.time()),
                            "event": "worker_timeout",
                            "assignment_id": assignment_id,
                            "elapsed_seconds": round(elapsed, 1),
                            "returncode": returncode,
                        },
                    )
                    continue
            if returncode is None:
                still_running.append((process, card, stdout_handle, stderr_handle, started_at))
                continue

            stdout_handle.close()
            stderr_handle.close()
            finished += 1
            if returncode != 0:
                failed += 1
            summary_path = Path(normalize_ws(card.get("summary_path")))
            if args.require_summary and not summary_path.exists():
                missing_summary += 1
                failed += 1
            append_jsonl(
                event_log,
                {
                    "ts": int(time.time()),
                    "event": "worker_finished",
                    "assignment_id": assignment_id,
                    "returncode": returncode,
                    "row_count": nonblank_line_count(Path(normalize_ws(card.get("output_path")))),
                    "summary_exists": summary_path.exists(),
                },
            )
        running = still_running

    summary = {
        "batch_id": args.batch_id,
        "dry_run": False,
        "assignment_count": len(cards),
        "started_count": started,
        "finished_count": finished,
        "timeout_count": timed_out,
        "failed_count": failed,
        "missing_summary_count": missing_summary,
        "parallel": args.parallel,
        "worker_timeout_seconds": args.worker_timeout_seconds,
        "artifacts": {
            "batch_dir": str(batch_dir),
            "events": str(event_log),
            "batch_summary": str(batch_dir / "batch_summary.json"),
        },
    }
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selected-assignments", type=Path, default=None)
    parser.add_argument("--assignment-id", action="append", default=[])
    parser.add_argument("--max-assignments", type=int, default=0)
    parser.add_argument("--batch-id", default=now_batch_id())
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=900)
    parser.add_argument("--worker-command", default=DEFAULT_WORKER_COMMAND)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--skip-completed",
        dest="skip_completed",
        action="store_true",
        help="Skip assignments that already have output rows and a summary file. This is the default.",
    )
    parser.add_argument(
        "--rerun-completed",
        dest="skip_completed",
        action="store_false",
        help="Intentionally rerun assignments even if they already look complete.",
    )
    parser.add_argument("--no-require-summary", dest="require_summary", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(require_summary=True, skip_completed=True)
    args = parser.parse_args()
    if args.selected_assignments is None:
        args.selected_assignments = args.output_root / "selected_assignments.jsonl"
    return args


def main() -> int:
    try:
        args = parse_args()
        cards = selected_cards(args)
        summary = run_cards(args, cards)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("failed_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
