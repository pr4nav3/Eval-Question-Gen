# Plan: Eval-Question-Gen

## Goal

Create a smaller eval set from:

```text
data/training_questions.csv
```

The eval rows must:

- use only chunks that were cited by training rows,
- ask new questions from the same training-seen facts,
- include paraphrases with different reasoning pressure, reverse lookups, applications, comparisons, combinations, easy recalls, hard synthesis, and fair traps,
- avoid copying the training questions or duplicate questions already written in the same assignment run,
- output the intentionally compact eval CSV schema.

The core principle:

```text
training rows show what the model already saw
exact cited chunks are the evidence boundary
clusters are candidate evidence neighborhoods, not semantic truth
Kimi performs local coherence checks, gap reasoning, and question writing
code handles hydration, bounded manifests, inside-run dedupe, validation, and export
```

## Non-Goals

- Do not create facts outside the exact cited chunks.
- Do not fetch neighboring chunks.
- Do not give Kimi the whole dataset or all hydrated chunks in prompt context.
- Do not force one eval row per training row or per cluster.
- Do not treat deterministic clusters as guaranteed semantic clusters.
- Do not hardcode API keys in source code.
- Do not add bookkeeping-only columns to the final eval CSV schema.

## Final Eval CSV Schema

```csv
question,answer,docIds,chunk_ids,pipeline
```

For generated eval rows:

| Column | Value |
|---|---|
| `question` | New eval question |
| `answer` | Gold answer grounded only in cited chunks |
| `docIds` | JSON array of cited doc IDs |
| `chunk_ids` | JSON array of exact training-seen chunk IDs |
| `pipeline` | `Eval-Question-Gen` |

## Active Pipeline

```text
source CSV
  -> delta train-key check
  -> source normalization
  -> deterministic metadata-only candidate cluster creation, optionally focused by delta keys
  -> bounded Kimi assignment selection
  -> assignment-local Vespa hydration of selected exact chunks
  -> bounded Kimi manifests plus per-assignment evidence bundles
  -> hands-free supervisor with max 3 total active agents
  -> Kimi coherence check + gap reasoning + row writing
  -> assignment-local collect + deterministic validation
  -> assignment-local fast LLM judge
  -> final CSV/export report plus compact eval bank append
```

## Phase 1: Source Normalization

Script:

```text
scripts/eval-question-gen/source_dataset.py
```

Responsibilities:

- parse source CSV,
- parse `docIds` and `chunk_ids` as JSON arrays,
- normalize chunk IDs into `doc_id` and `chunk_index`,
- keep duplicate rows for statistics but exclude exact duplicates from generation pressure,
- build row/chunk/doc/evidence-signature indexes.

Artifact:

```text
runs/<run_id>/phase_source_summary.json
```

## Phase 2: Metadata-Only Candidate Clustering

Script:

```text
scripts/eval-question-gen/cluster_evidence.py
```

Create candidate evidence neighborhoods before hydrating chunk text. This phase
uses only metadata already present in the training CSV:

- `docIds`,
- `chunk_ids`,
- chunk indexes parsed from IDs such as `doc_id#17`,
- source rows that cite each chunk,
- chunks cited together by the same training row,
- docs cited together by the same training row.

No Vespa fetch is required in the normal path.

Cluster kinds:

| Kind | Meaning | Trust Level |
|---|---|---|
| `exact_evidence_set` | Same sorted chunk set cited by one or more training rows | Highest |
| `doc_set` | Rows cite the same sorted doc ID set | Medium |
| `co_citation` | Chunks connected because training rows cited them together | High/medium |
| `doc_local_seen_chunks` | Training-seen chunks close by index in the same doc | Medium/risky |
| `multi_doc_bridge` | Multi-doc groupings inferred from citation overlap | Useful but risky |

Cluster controls:

```text
max_chunks_per_cluster: bounded candidate envelope
doc_local_max_gap: maximum chunk-index gap for doc-local grouping
limit_clusters: optional smoke-test limit
```

Clusters are candidate work units, not semantic truth. Kimi still decides whether
the hydrated cluster is coherent enough to produce eval rows.

Artifacts:

```text
runs/<run_id>/clusters.jsonl
runs/<run_id>/cluster_summary.json
```

## No Full-Dataset Hydration

Full-dataset hydration is not part of this pipeline. The scripts must never
hydrate every training-seen chunk up front.

Allowed hydration:

- selected assignment chunks,
- specific cluster chunks being audited,
- only exact `chunk_ids` from the source CSV.

Not allowed:

- hydrating the whole source CSV,
- hydrating neighboring chunks,
- using raw PDFs, graph neighbors, corpus search, or external sources.

Any audit must be cluster-local: hydrate only the exact chunks for the cluster
or generated question being audited, then compare against the complete related
training rows for that same cluster/evidence neighborhood.

## Phase 3: Inside-Run Dedupe

Scripts:

```text
scripts/eval-question-gen/collect_kimi_eval_outputs.py
scripts/eval-question-gen/prepare_kimi_eval_assignments.py
```

Purpose:

Prevent Kimi from spending compute recreating similar eval questions inside the
same assignment run, without showing it old eval questions from unrelated runs.

Run-local ledger:

```text
assignments/<assignment_run>/run_memory/generated_eval_rows.jsonl
```

The ledger is append-only and compact for that assignment run only. It stores
generated eval rows with enough metadata to retrieve them by document key:

```json
{
  "created_at": 1234567890,
  "run_dir": "runs/...",
  "assignment_id": "kimi_eval_000007",
  "cluster_id": "eqg_cocite_006462",
  "microcluster_id": "eqg_cocite_006462",
  "cluster_kind": "co_citation",
  "doc_key": "hash-of-sorted-docIds",
  "allowed_docIds": ["clf-a"],
  "used_chunk_ids": ["clf-a#1"],
  "question": "...",
  "eval_question_hash": "hash-of-normalized-eval-question",
  "answer": "...",
  "seed_train_keys": ["hash-of-training-row"],
  "status": "generated"
}
```

Assignment prep loads this run-local ledger and adds only a small, overlapping
same-run dedupe list to each Kimi assignment:

```json
"run_previous_questions_to_avoid": [
  {
    "question": "...",
    "answer_short": "...",
    "used_chunk_ids": ["..."],
    "assignment_id": "...",
    "status": "generated"
  }
]
```

Selection rule:

- include same-run rows with the same `doc_key`,
- cap the memory block, default `5` rows,
- never show cross-run eval history to Kimi.

Important: same-run previous questions are dedupe hints, not evidence. Kimi may
use them only to avoid repeated question ideas.

## Phase 4: Assignment-Local Hydration And Kimi Manifests

Script:

```text
scripts/eval-question-gen/prepare_kimi_eval_assignments.py
```

Assignment prep selects bounded metadata clusters, hydrates only the selected
cluster chunks from Vespa, and writes one small evidence bundle per assignment.

Each assignment manifest includes:

- `allowed_chunk_ids`,
- compact chunk references,
- doc metadata,
- cluster kind/provenance/support,
- complete related training row numbers for the cluster/evidence neighborhood,
- assignment-local JSONL file containing a capped, relevance-ranked visible slice of related training rows,
- same-run previous questions to avoid, if any,
- artifact paths for assignment-local training rows and hydrated evidence bundles,
- output path and summary path.

Kimi reads exact chunk text from the assignment-local evidence bundle:

```text
artifact_paths.assignment_hydrated_chunks_jsonl
```

Default assignment sizing:

```text
target_chunks_per_assignment: 2-6
max_chunks_per_assignment: 8
training_rows_preview_in_manifest: 0
max_run_previous_questions: 5
max_eval_rows_per_assignment: 2
max_manifest_chars: 30000, excluding assignment-local chunk text bundle
```

If a selected assignment cannot hydrate all of its exact chunks, assignment prep
rejects it before Kimi sees it.

Related training row invariant:

- start from the cluster's own support rows,
- add every source row that cites any `allowed_chunk_ids`,
- keep the complete stable set in `seed_train_keys`,
- write the capped, relevance-ranked visible related row objects to
  `artifact_paths.related_training_rows_jsonl`.

This means the manifest never loses the complete related-row identity set, even
though no training questions are embedded in the assignment JSON. Kimi reads the
visible training-row bundle before gap finding. If the visible row count looks
mismatched, it notes that once in the summary and continues when the evidence is
usable.

## Phase 5: Kimi Worker Prompt

Prompt file:

```text
scripts/eval-question-gen/KIMI_EVAL_AGENT_INSTRUCTIONS.md
```

Kimi must:

1. Read related training rows.
2. Read same-run previous questions for this cluster/evidence, if present.
3. Inspect only allowed chunks from the assignment evidence bundle.
4. Decide whether the candidate cluster is coherent.
5. Find gaps in question reasoning:
   - paraphrase with different reasoning pressure,
   - reverse lookup,
   - application,
   - comparison,
   - combination of facts,
   - unused fact,
   - rule/condition/exception,
   - fair trap directly refuted by chunks.
6. Write zero to two rows.
7. Write a concise summary with reasoning gaps used, packet status, and optional one-line issue notes.

Kimi must not:

- cite chunks outside `allowed_chunk_ids`,
- use same-run previous questions as evidence,
- load unrelated evidence bundles,
- search neighboring chunks,
- search the corpus,
- use raw PDFs,
- use graph neighbors or external knowledge.

## Phase 6: Kimi Runner

Script:

```text
scripts/eval-question-gen/run_kimi_eval_assignments.py
```

Responsibilities:

- load `.env`,
- export the same key as `LITELLM_API_KEY`, `JUSPAY_API_KEY`, and `OPENAI_API_KEY` for OpenCode/provider compatibility,
- launch OpenCode workers with `litellm/private-large`,
- enforce worker timeout,
- write per-worker stdout/stderr logs,
- write batch events and summary.

Default worker command:

```text
opencode run -m litellm/private-large "Read and execute the run prompt at {prompt_path}"
```

Default timeout:

```text
900 seconds
```

Artifacts:

```text
assignments/<run>/batch_runs/<batch_id>/events.jsonl
assignments/<run>/batch_runs/<batch_id>/batch_summary.json
```

## Phase 7: Hands-Free Supervisor

Script:

```text
scripts/eval-question-gen/run_eval_supervisor.py
```

This is the normal hands-free path after assignment prep. It reuses the prepared
assignments and keeps one global agent budget across both Kimi generation
workers and LLM judge workers.

Default active-agent limit:

```text
max_active_agents: 3
```

Supervisor loop:

1. Recover existing assignment outputs and skip already completed work.
2. Start pending Kimi generation workers while slots are available.
3. When an assignment finishes, collect and validate only that assignment's rows.
4. If rows pass validation, queue an assignment-scoped judge for those rows.
5. Prioritize pending judges before starting more generation.
6. Stop launching new generation once the target accepted-row count is reached.
7. Finish judging any already-generated assignment backlog before export.
8. Export only from `judge_accepted_candidates.jsonl`.

Supervisor artifacts:

```text
runs/<run_id>/supervisor/supervisor_state.json
runs/<run_id>/supervisor/status.md
runs/<run_id>/supervisor/events.jsonl
runs/<run_id>/supervisor/assignments/<assignment_id>/
runs/<run_id>/supervisor/logs/
```

The supervisor also rebuilds the normal run-level artifacts after each state
transition, so downstream validation/export files keep their existing names.

## Phase 8: Collect Kimi Outputs

Script:

```text
scripts/eval-question-gen/collect_kimi_eval_outputs.py
```

Responsibilities:

- read assignment outputs,
- reject malformed rows,
- enforce cited chunks are a subset of that assignment's `allowed_chunk_ids`,
- reject missing or mismatched `docIds` instead of correcting them silently,
- normalize top-level CSV fields,
- write `generated_candidates.jsonl` and `generated_candidates.csv`,
- append generated rows to the run-local dedupe ledger.

Artifacts:

```text
runs/<run_id>/generated_candidates.jsonl
runs/<run_id>/generated_candidates.csv
runs/<run_id>/generation_summary.json
runs/<run_id>/kimi_collection_summary.json
assignments/<assignment_run>/run_memory/generated_eval_rows.jsonl
```

## Phase 9: Validation

Script:

```text
scripts/eval-question-gen/validate_rows.py
```

Reject if:

- schema fields are missing,
- `docIds` or `chunk_ids` are not arrays,
- any chunk was not in the source CSV's training-seen chunk set,
- any chunk is outside the assignment's allowed chunk set,
- any chunk is outside the candidate cluster's chunk set,
- generated `docIds` do not match generated `chunk_ids`,
- question or answer is empty,
- question contains internal chunk IDs,
- normalized question exactly matches a training question,
- question is too similar to training questions for the complete related
  training-row set carried by the assignment.

Artifacts:

```text
runs/<run_id>/validated_candidates.jsonl
runs/<run_id>/validation_accepted_candidates.jsonl
runs/<run_id>/validation_rejected_candidates.jsonl
runs/<run_id>/validation_summary.json
```

## Phase 10: Fast Cluster-Local Judge

Script:

```text
scripts/eval-question-gen/judge_eval_rows.py
```

Run after deterministic validation and before export. The judge is intentionally
small: one candidate row at a time, using only the candidate's cited chunks from
the assignment-local evidence bundle.

The judge receives:

- generated question and answer,
- cited chunks only,
- assignment and cluster provenance,
- a compact relevance-sorted subset of related training rows for distinctness,
- small same-run previous-question hints for the same or overlapping evidence.

The full related training-row bundle must exist for the assignment. If it is
missing, the candidate is rejected before any LLM call.

The judge keeps output compact. It should internally check answer support, but
only return the fields needed for gating: answer support, distinctness,
citation quality, eval quality, supporting chunk IDs, similar refs, and one
short reason.

The judge should reject rows that:

- require unseen context,
- copy or lightly reword a training/eval question,
- cite unnecessary chunks,
- use unsupported facts,
- make unfair traps,
- are too boilerplate-heavy.

Artifacts:

```text
runs/<run_id>/judge_candidates.jsonl
runs/<run_id>/judge_accepted_candidates.jsonl
runs/<run_id>/judge_rejected_candidates.jsonl
runs/<run_id>/judge_errors.jsonl
runs/<run_id>/judge_summary.json
```

## Phase 11: Export

Script:

```text
scripts/eval-question-gen/export_results.py
```

Final artifacts:

```text
runs/<run_id>/eval_seen_chunks.csv
runs/<run_id>/eval_seen_chunks.jsonl
runs/<run_id>/run_report.md
runs/<run_id>/export_summary.json
```

The final CSV must contain only the eval schema columns listed above. The
training source CSV may include bookkeeping columns such as `source_row_number`,
`is_exact_duplicate`, and `duplicate_of_row`; those are not emitted in final
eval CSVs.

Export requires a compatible `judge_summary.json` and reads only
`judge_accepted_candidates.jsonl`. It must fail closed if the judge has not run.

## Current Implementation Files

| File | Active Role |
|---|---|
| `create_eval_questions.py` | Orchestrates source, cluster, validate, and export phases |
| `pipeline_paths.py` | Standalone repo-root default paths |
| `training_delta.py` | Computes new `train_key` values and writes delta focus keys |
| `source_dataset.py` | Source CSV parsing and normalization |
| `vespa_chunks.py` | Exact selected-doc/chunk hydration helpers for assignments and cluster-local audits |
| `cluster_evidence.py` | Deterministic candidate cluster construction |
| `prepare_kimi_eval_assignments.py` | Bounded Kimi manifests with related training rows and run-local dedupe hints |
| `KIMI_EVAL_AGENT_INSTRUCTIONS.md` | Worker instructions for coherence, gap reasoning, and row writing |
| `run_kimi_eval_assignments.py` | OpenCode/Kimi worker launcher with `.env` loading |
| `run_eval_supervisor.py` | Hands-free supervisor with worker->judge queue and max 3 active agents |
| `collect_kimi_eval_outputs.py` | Collects Kimi outputs and updates run-local dedupe |
| `validate_rows.py` | Deterministic validation |
| `judge_eval_rows.py` | Fast cluster-local LLM quality judge |
| `export_results.py` | Final CSV/JSONL/report export and eval-bank append |
| `export_eval_subset.py` | Exports eval-bank rows relevant to a subset training CSV |

## Standard Commands

Source and cluster setup:

Delta check:

```bash
python3 scripts/eval-question-gen/training_delta.py \
  --input data/training_questions.csv \
  --snapshot-id <snapshot_id>
```

```bash
python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase source \
  --run-id <run_id> \
  --env-file .env

python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase cluster \
  --run-id <run_id> \
  --env-file .env \
  --max-chunks-per-cluster 8 \
  --doc-local-max-gap 3 \
  --focus-train-keys deltas/<snapshot_id>/delta_train_keys.jsonl
```

Prepare Kimi assignments:

```bash
python3 scripts/eval-question-gen/prepare_kimi_eval_assignments.py \
  --run-dir runs/<run_id> \
  --output-root assignments/<assignment_run> \
  --target-rows 200 \
  --questions-per-assignment 2 \
  --min-chunks 2 \
  --max-chunks 6 \
  --max-run-previous-questions 5
```

For exploratory one-cluster checks where Kimi should choose the row count
naturally, add:

```bash
--natural-question-count
```

Run hands-free generation, validation, judging, and export:

```bash
python3 scripts/eval-question-gen/run_eval_supervisor.py run \
  --input data/training_questions.csv \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run> \
  --target-rows 200 \
  --max-active-agents 3 \
  --llm-model private-large \
  --env-file .env
```

Check supervisor status:

```bash
python3 scripts/eval-question-gen/run_eval_supervisor.py status \
  --run-dir runs/<run_id>
```

Manual fallback: run Kimi batch only:

```bash
python3 scripts/eval-question-gen/run_kimi_eval_assignments.py \
  --output-root assignments/<assignment_run> \
  --parallel 2 \
  --worker-timeout-seconds 900 \
  --batch-id <batch_id>
```

Manual fallback: collect, validate, judge, and export:

```bash
python3 scripts/eval-question-gen/collect_kimi_eval_outputs.py \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run>

python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase validate \
  --run-id <run_id>

python3 scripts/eval-question-gen/judge_eval_rows.py \
  --run-dir runs/<run_id> \
  --assignment-root assignments/<assignment_run> \
  --env-file .env \
  --workers 2

python3 scripts/eval-question-gen/create_eval_questions.py \
  --phase export \
  --run-id <run_id>
```

## Quality Expectations

Good eval rows should:

- be answerable from cited chunks only,
- use facts the model saw in training through those chunks,
- differ in reasoning operation from the training questions,
- avoid duplicate questions already written in the same assignment run,
- be specific enough to grade,
- avoid overusing boilerplate recovery-payment/statutory-authority questions.

Good Kimi summaries should state:

- reasoning gaps used,
- packet status: `saturated`, `weak`, or `usable`,
- optional one-line issue notes.

## Known Operational Reality

The mid-sized Kimi test showed:

- API key loading from `.env` worked.
- Kimi produced structurally valid rows from bounded manifests.
- Successful rows passed deterministic validation.
- Some workers timed out or failed inside OpenCode/model execution.

This means the pipeline shape is viable, but full hands-free generation should over-sample assignments and expect some worker failures.
