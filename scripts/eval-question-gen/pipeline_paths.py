#!/usr/bin/env python3
"""Repository-local default paths for Eval-Question-Gen."""

from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INPUT = REPO_ROOT / "data" / "training_questions.csv"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_RUN_DIR = DEFAULT_RUNS_DIR / "eval_metadata_clusters"
DEFAULT_ASSIGNMENT_ROOT = REPO_ROOT / "assignments"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_EVAL_BANK = REPO_ROOT / "bookkeeping" / "eval_bank.jsonl"
DEFAULT_TRAINING_STATE = REPO_ROOT / "bookkeeping" / "training_row_state.jsonl"
DEFAULT_DELTAS_ROOT = REPO_ROOT / "deltas"
