# Eval Question Gen Runbook

All commands assume they are run from the repository root.

## 1. Configure

```bash
cp .env.example .env
```

Set:

- `LITELLM_BASE_URL`
- `LITELLM_API_KEY`
- `EVAL_LLM_MODEL=private-large`
- `VESPA_QUERY_URL`

Put the merged training CSV here:

```text
data/training_questions.csv
```

## 2. Optional Delta Check

```bash
python3 scripts/eval-question-gen/training_delta.py \
  --input data/training_questions.csv \
  --snapshot-id <snapshot_id>
```

This writes:

```text
deltas/<snapshot_id>/delta_train_keys.jsonl
```

## 3. Source And Cluster

```bash
python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase cluster \
  --run-id <run_id> \
  --max-chunks-per-cluster 8 \
  --doc-local-max-gap 3
```

To focus on a delta:

```bash
python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase cluster \
  --run-id <run_id> \
  --max-chunks-per-cluster 8 \
  --doc-local-max-gap 3 \
  --focus-train-keys deltas/<snapshot_id>/delta_train_keys.jsonl
```

## 4. Prepare Assignments

```bash
python3 scripts/eval-question-gen/prepare_kimi_eval_assignments.py \
  --run-dir runs/<run_id> \
  --output-root assignments/<assignment_run> \
  --target-rows 200 \
  --questions-per-assignment 2 \
  --min-chunks 2 \
  --max-chunks 6 \
  --max-related-training-rows 200 \
  --max-run-previous-questions 5
```

This hydrates only selected assignment chunks from Vespa.

## 5. Run Hands-Free Supervisor

```bash
python3 scripts/eval-question-gen/run_eval_supervisor.py run \
  --input data/training_questions.csv \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run> \
  --target-rows 200 \
  --max-active-agents 3 \
  --llm-model private-large
```

Check status:

```bash
python3 scripts/eval-question-gen/run_eval_supervisor.py status \
  --run-dir runs/<run_id>
```

## 6. Manual Fallback

Generate:

```bash
python3 scripts/eval-question-gen/run_kimi_eval_assignments.py \
  --output-root assignments/<assignment_run> \
  --parallel 2 \
  --worker-timeout-seconds 900 \
  --batch-id <batch_id>
```

Collect:

```bash
python3 scripts/eval-question-gen/collect_kimi_eval_outputs.py \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run>
```

Validate:

```bash
python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase validate \
  --run-id <run_id>
```

Judge:

```bash
python3 scripts/eval-question-gen/judge_eval_rows.py \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run> \
  --llm-model private-large \
  --workers 2
```

Export:

```bash
python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase export \
  --run-id <run_id>
```

Export fails closed unless compatible judge outputs exist.
