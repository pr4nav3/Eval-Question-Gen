#!/usr/bin/env python3
"""Eval-Question-Gen orchestrator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cluster_evidence import build_seen_evidence_clusters
from export_results import export_final_eval
from llm_client import apply_env_file
from pipeline_paths import DEFAULT_ENV_FILE, DEFAULT_EVAL_BANK, DEFAULT_INPUT, DEFAULT_RUNS_DIR
from source_dataset import (
    create_source_analysis,
    load_source_rows,
    write_normalized_rows_jsonl,
    write_source_analysis_json,
)
from validate_rows import validate_generated_rows


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_run_dir(output_dir: Path, run_id: str) -> Path:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_focus_train_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stripped = line.strip()
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                keys.add(stripped)
                continue
            if isinstance(parsed, dict):
                value = parsed.get("train_key")
                if value:
                    keys.add(str(value))
            elif isinstance(parsed, str):
                keys.add(parsed)
    return keys


def run_source_phase(input_path: Path, run_dir: Path) -> dict[str, Any]:
    rows, issues, fieldnames = load_source_rows(input_path)
    analysis = create_source_analysis(input_path, rows, issues, fieldnames)

    normalized_rows_path = run_dir / "normalized_rows.jsonl"
    source_analysis_path = run_dir / "source_analysis.json"
    phase_summary_path = run_dir / "phase_source_summary.json"

    write_normalized_rows_jsonl(rows, normalized_rows_path)
    write_source_analysis_json(analysis, source_analysis_path)

    summary = {
        "phase": "source",
        "input_path": str(input_path),
        "run_dir": str(run_dir),
        "artifacts": {
            "source_analysis": str(source_analysis_path),
            "normalized_rows": str(normalized_rows_path),
        },
        "row_counts": analysis["row_counts"],
        "unique_counts": analysis["unique_counts"],
        "parse_issue_count": analysis["integrity"]["parse_issue_count"],
    }
    with phase_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary["artifacts"]["phase_summary"] = str(phase_summary_path)
    return summary


def run_cluster_phase(
    input_path: Path,
    run_dir: Path,
    *,
    max_chunks_per_cluster: int,
    doc_local_max_gap: int,
    limit_clusters: int | None,
    focus_train_keys_path: Path | None,
) -> dict[str, Any]:
    rows, _, _ = load_source_rows(input_path)
    return build_seen_evidence_clusters(
        rows,
        run_dir=run_dir,
        max_chunks_per_cluster=max_chunks_per_cluster,
        doc_local_max_gap=doc_local_max_gap,
        limit_clusters=limit_clusters,
        focus_train_keys=load_focus_train_keys(focus_train_keys_path),
    )


def run_validate_phase(
    input_path: Path,
    run_dir: Path,
    *,
    similarity_threshold: float,
) -> dict[str, Any]:
    rows, _, _ = load_source_rows(input_path)
    return validate_generated_rows(
        rows,
        run_dir=run_dir,
        similarity_threshold=similarity_threshold,
    )


def run_export_phase(run_dir: Path, *, eval_bank_path: Path) -> dict[str, Any]:
    return export_final_eval(run_dir=run_dir, eval_bank_path=eval_bank_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a seen-chunk eval set from the merged training CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source merged CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Run output root. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--run-id",
        default=default_run_id(),
        help="Run ID used as the output subdirectory name.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f"Env file loaded before running phases. Default: {DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--phase",
        choices=["source", "cluster", "validate", "export"],
        default="source",
        help="Pipeline phase for the Kimi-first flow. Hydration happens only in prepare_kimi_eval_assignments.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and analyze but do not write artifacts.",
    )
    parser.add_argument(
        "--max-chunks-per-cluster",
        type=int,
        default=30,
        help="Maximum chunk IDs in a metadata-only cluster envelope.",
    )
    parser.add_argument(
        "--doc-local-max-gap",
        type=int,
        default=3,
        help="Maximum seen chunk-index gap for doc-local clusters.",
    )
    parser.add_argument(
        "--limit-clusters",
        type=int,
        default=None,
        help="Write only the first N clusters. Useful for smoke tests.",
    )
    parser.add_argument(
        "--focus-train-keys",
        type=Path,
        default=None,
        help="Optional JSONL/text file of train_key values; cluster output is limited to clusters touched by these keys.",
    )
    parser.add_argument(
        "--eval-bank-path",
        type=Path,
        default=DEFAULT_EVAL_BANK,
        help=f"Accepted eval bank JSONL. Default: {DEFAULT_EVAL_BANK}",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Token Jaccard threshold for rejecting near-training-question duplicates.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.env_file:
        apply_env_file(str(args.env_file))
    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    if args.dry_run:
        rows, issues, fieldnames = load_source_rows(input_path)
        analysis = create_source_analysis(input_path, rows, issues, fieldnames)
        print(
            json.dumps(
                {
                    "phase": "source",
                    "dry_run": True,
                    "row_counts": analysis["row_counts"],
                    "unique_counts": analysis["unique_counts"],
                    "parse_issue_count": analysis["integrity"]["parse_issue_count"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    run_dir = ensure_run_dir(args.output_dir, args.run_id)
    summaries: list[dict[str, Any]] = []

    if args.phase in {"source", "cluster"}:
        summaries.append(run_source_phase(input_path, run_dir))
    if args.phase == "cluster":
        summaries.append(
            run_cluster_phase(
                input_path,
                run_dir,
                max_chunks_per_cluster=args.max_chunks_per_cluster,
                doc_local_max_gap=args.doc_local_max_gap,
                limit_clusters=args.limit_clusters,
                focus_train_keys_path=args.focus_train_keys,
            )
        )
    if args.phase == "validate":
        summaries.append(
            run_validate_phase(
                input_path,
                run_dir,
                similarity_threshold=args.similarity_threshold,
            )
        )
    if args.phase == "export":
        summaries.append(run_export_phase(run_dir, eval_bank_path=args.eval_bank_path))
    print(json.dumps({"run_dir": str(run_dir), "phases": summaries}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
