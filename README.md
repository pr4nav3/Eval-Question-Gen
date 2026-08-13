# Eval Question Gen

Standalone workflow for generating evaluation questions from training-seen evidence chunks.

The pipeline reads a merged training-question CSV, clusters only the cited chunk IDs, hydrates selected assignment chunks from local Vespa, asks Kimi/OpenCode to write new questions, validates the rows, judges them with `private-large`, and exports a compact eval CSV.

## Requirements

- Python 3.10+.
- `opencode` CLI available on `PATH`.
- A LiteLLM/OpenAI-compatible endpoint with `private-large`.
- Local Vespa query endpoint exposing `kb_items` with `docId`, `chunks_summary`, and `chunks_map`.
- A training CSV with columns:

```csv
question,answer,docIds,chunk_ids,pipeline,source_row_number,is_exact_duplicate,duplicate_of_row
```

Final eval CSVs intentionally contain only:

```csv
question,answer,docIds,chunk_ids,pipeline
```

## Repo Layout

```text
data/training_questions.csv      # local input, not committed
runs/                            # run artifacts, not committed
assignments/                     # Kimi assignment artifacts, not committed
bookkeeping/                     # eval bank and delta state, not committed
deltas/                          # training delta outputs, not committed
scripts/eval-question-gen/       # pipeline code and prompts
.env                             # local secrets/config, not committed
```

## Quick Start

1. Copy `.env.example` to `.env` and fill in local values.
2. Place your merged training CSV at `data/training_questions.csv`.
3. Run the workflow in the order shown in `RUNBOOK.md`.

The normal hands-free path is:

```bash
python3 scripts/eval-question-gen/create_eval_questions.py --phase cluster --run-id <run_id>

python3 scripts/eval-question-gen/prepare_kimi_eval_assignments.py \
  --run-dir runs/<run_id> \
  --output-root assignments/<assignment_run> \
  --target-rows 200

python3 scripts/eval-question-gen/run_eval_supervisor.py run \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run> \
  --target-rows 200
```

The final CSV is written to:

```text
runs/<run_id>/eval_seen_chunks.csv
```
