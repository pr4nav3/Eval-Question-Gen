# Eval Question Gen Setup Instructions

This repo is standalone code, but it still needs local runtime services and
local input files. Nothing sensitive or large should be committed.

All commands below assume you are running from this repository root:

```bash
cd "/path/to/Eval Question Gen"
```

## Required Local Inputs

- Training CSV: put the merged training set at `data/training_questions.csv`,
  or pass `--input /path/to/file.csv` to each command that reads training rows.
- Env file: copy `.env.example` to `.env` and fill in local values.
- Local Vespa: the pipeline hydrates selected chunks from a local Vespa query
  endpoint. This is a URL, not a repo path, and it may differ by machine.
- LiteLLM/OpenAI-compatible endpoint: must expose `private-large`.
- `opencode` CLI: must be available on `PATH`.

The training CSV must contain these columns:

```csv
question,answer,docIds,chunk_ids,pipeline,source_row_number,is_exact_duplicate,duplicate_of_row
```

The final eval CSV intentionally contains only:

```csv
question,answer,docIds,chunk_ids,pipeline
```

Do not commit local runtime data or secrets:

```text
.env
data/
runs/
assignments/
bookkeeping/
deltas/
```

## `.env` Values

Minimum required values:

```bash
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=<local-key>
EVAL_LLM_MODEL=private-large
VESPA_QUERY_URL=http://localhost:18081/search/
```

`VESPA_QUERY_URL` must point at the Vespa query service that contains `kb_items`
with `docId`, `chunks_summary`, and `chunks_map`. The correct host port is
machine-specific:

- Docker mapping like `0.0.0.0:18081->8081/tcp`: use
  `http://localhost:18081/search/`.
- Native/local Vespa directly on 8081: use `http://localhost:8081/search/`.

Verify the endpoint before running generation:

```bash
curl -sS -i --get "$VESPA_QUERY_URL" \
  --data-urlencode 'yql=select docId from kb_items limit 1' \
  --data-urlencode hits=1
```

A healthy endpoint returns HTTP 200 and either a hit or an empty result. HTTP
503 with a backend communication error means that endpoint is reachable but the
Vespa content backend behind it is not healthy. Pick the correct port or fix
the Vespa service before running assignment preparation.

## Preflight Checks

Check Python:

```bash
python3 --version
python3 -m py_compile scripts/eval-question-gen/*.py
```

Check OpenCode:

```bash
command -v opencode
opencode --help
```

Check the LiteLLM/private-large endpoint:

```bash
curl -sS "$LITELLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "private-large",
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 8
  }'
```

If your local LiteLLM base URL already includes `/v1`, use
`$LITELLM_BASE_URL/chat/completions`. If it does not, this repo's Python judge
client handles that automatically for judge calls, but the manual `curl` should
target the actual chat-completions URL exposed by your local service.

## Normal Run

```bash
python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase cluster \
  --run-id <run_id>

python3 scripts/eval-question-gen/prepare_kimi_eval_assignments.py \
  --run-dir runs/<run_id> \
  --output-root assignments/<assignment_run> \
  --target-rows 200 \
  --questions-per-assignment 2

python3 scripts/eval-question-gen/run_eval_supervisor.py run \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run> \
  --target-rows 200 \
  --llm-model private-large
```

The final export, when enough rows pass the judge, is:

```text
runs/<run_id>/eval_seen_chunks.csv
```

Common output locations:

```text
runs/<run_id>/clusters.jsonl
runs/<run_id>/cluster_summary.json
assignments/<assignment_run>/selected_assignments.jsonl
assignments/<assignment_run>/assignment_summary.json
runs/<run_id>/supervisor/status.md
runs/<run_id>/judge_summary.json
runs/<run_id>/eval_seen_chunks.csv
bookkeeping/eval_bank.jsonl
```

Export fails closed unless compatible judge outputs exist. It reads only
judge-accepted rows and writes the compact schema:

```csv
question,answer,docIds,chunk_ids,pipeline
```

## Prompt Policy

There are two prompt layers.

Kimi generation prompt:

- Default: `scripts/eval-question-gen/KIMI_EVAL_AGENT_INSTRUCTIONS_V3.md`.
  This asks Kimi for reasoning-heavy questions and is the recommended default.
- Older alternatives:
  `KIMI_EVAL_AGENT_INSTRUCTIONS.md` and
  `KIMI_EVAL_AGENT_INSTRUCTIONS_V2.md`.
- Override during assignment preparation with:

```bash
python3 scripts/eval-question-gen/prepare_kimi_eval_assignments.py \
  --agent-instructions scripts/eval-question-gen/KIMI_EVAL_AGENT_INSTRUCTIONS_V3.md \
  ...
```

Judge prompt and difficulty gate:

- Default judge policy requires `medium` or `hard` rows. The deterministic
  Python gate enforces this with `--min-difficulty medium`.
- Strict external prompt:
  `scripts/eval-question-gen/JUDGE_SYSTEM_PROMPT_V3.md`.
- Lower-difficulty fallback:
  `scripts/eval-question-gen/JUDGE_SYSTEM_PROMPT_EASY_OK.md`.

If useful supported rows are being rejected only as `difficulty_too_low`, either
improve the Kimi prompt/instructions first, or intentionally allow easier recall
eval rows with:

```bash
python3 scripts/eval-question-gen/run_eval_supervisor.py run \
  --judge-system-prompt scripts/eval-question-gen/JUDGE_SYSTEM_PROMPT_EASY_OK.md \
  --judge-min-difficulty easy \
  ...
```

For manual judge runs, use:

```bash
python3 scripts/eval-question-gen/judge_eval_rows.py \
  --system-prompt scripts/eval-question-gen/JUDGE_SYSTEM_PROMPT_EASY_OK.md \
  --min-difficulty easy \
  ...
```

The lower-difficulty path should be a conscious eval-design choice, not a
silent fallback.

## Throughput Expectations

The isolated repo uses the same core runtime path as the original pipeline:
metadata clustering, selected-cluster hydration, `opencode run -m
litellm/private-large` generation, deterministic validation, LLM judge, and
export. The standalone version does not add an extra LLM stage.

Slow runs usually come from:

- `private-large` latency inside OpenCode generation workers;
- judge calls waiting on the same model;
- too-low `--max-active-agents`;
- unhealthy or wrong Vespa endpoint during hydration.

Useful knobs:

```bash
--max-active-agents 3
--worker-timeout-seconds 1200
--judge-timeout-seconds 900
--judge-llm-timeout-seconds 180
--max-worker-retries 1
--max-judge-retries 2
```

When comparing against the original pipeline, use the same model, same Vespa
endpoint, same `opencode` CLI, same concurrency, and same prompt files.

## Sanity Checks

Before a production run:

```bash
python3 -m py_compile scripts/eval-question-gen/*.py
python3 scripts/eval-question-gen/create_eval_questions.py --help
python3 scripts/eval-question-gen/run_eval_supervisor.py --help
```

After assignment preparation:

- `assignment_summary.json` should show `selected_assignment_count > 0`.
- `assignment_hydration.status_counts` should be `{"ok": ...}`.
- `hydration_rejected_assignment_count` should be `0` for selected work.

After supervisor:

- `generated_count` shows Kimi produced rows.
- `validation_ok_count` shows deterministic schema/evidence checks passed.
- `judge_accepted_count` controls whether export can happen.
- `export_summary.json` appears only after accepted judge rows are exported.

## Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Vespa query returns HTTP 503 with backend communication error | Wrong Vespa port or unhealthy content node | Verify `VESPA_QUERY_URL`; check Docker port mapping; use the healthy query port. |
| `assignment_summary.json` has `selected_assignment_count: 0` | Hydration failed or filters are too strict | Check `assignment_hydration.status_counts`; confirm Vespa has the cited docs; lower `--min-chunks` only if intentional. |
| `hydration_rejected_assignment_count` is high | Chunks were missing or Vespa fetch failed | Inspect `assignment_hydration.jsonl` and fix the endpoint/service before generation. |
| Kimi workers are slow | `private-large` or OpenCode latency | Increase `--max-active-agents`, keep timeouts realistic, and compare against the original pipeline with the same model/concurrency. |
| Kimi worker exits with timeout | OpenCode/model call timed out | Let the supervisor retry, or raise `--worker-timeout-seconds` if the model is consistently slow. |
| `validation_ok_count` is lower than `generated_count` | Kimi produced malformed rows or cited invalid chunks | Inspect assignment validation summaries and generation logs. |
| `judge_accepted_count` is zero but rows are supported | Judge difficulty gate may be too strict for this eval target | Prefer stronger Kimi instructions first; if easy recall rows are acceptable, use `JUDGE_SYSTEM_PROMPT_EASY_OK.md` with `--judge-min-difficulty easy`. |
| Export does not appear | No judge-accepted rows yet, or judge outputs are missing/incompatible | Check `judge_summary.json`; export intentionally fails closed. |
