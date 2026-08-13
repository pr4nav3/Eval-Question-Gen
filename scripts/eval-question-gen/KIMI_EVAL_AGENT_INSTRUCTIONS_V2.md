# Kimi Seen-Chunk Eval Instructions

You are creating eval questions from training-seen evidence.

## Contract

- Use only the exact chunks listed in `allowed_chunk_ids`.
- Read chunk text from `artifact_paths.assignment_hydrated_chunks_jsonl`.
- Do not load unrelated evidence files; inspect only the assignment-local chunk evidence.
- Do not search neighboring chunks, raw PDFs, graph neighbors, or external sources.
- Do not create one eval row per training row.
- Treat each assignment as a candidate cluster. It may be coherent, partially coherent, or noisy.
- Create only questions that are meaningfully different from the related training questions in the assignment.
- `seed_train_keys` is the complete stable related training-row provenance set for this assignment.
- `artifact_paths.related_training_rows_jsonl` contains the capped, relevance-ranked visible related training rows for this assignment.
- The assignment JSON intentionally does not embed training question previews.
- Read every JSONL object in `artifact_paths.related_training_rows_jsonl` before gap finding.
- If the visible related row count appears mismatched against `related_training_rows.artifact_row_count`, mention it once in the summary and continue if the evidence is usable.
- New rows may be paraphrases with different reasoning pressure, reverse lookups, applications, comparisons, or combinations of the same facts the model saw during training through these exact chunks.
- Do not duplicate questions already written in this assignment run.
- If `run_previous_questions_to_avoid` is present, use it only as same-run dedupe context, not as evidence.
- Prefer 1-3 strong rows over many thin rewrites.
- Each output row must be fully answerable from its cited chunk IDs.
- The cited chunk IDs must be a subset of `allowed_chunk_ids`.
- The cited doc IDs must match the cited chunk IDs.
- Never mention internal chunk IDs, cluster IDs, or assignment IDs in the question.

## Question style

A strong eval question makes the model connect facts, not just locate them.
Prefer questions whose answers require one of the following:

- Explaining *why* a regulator or court reached a conclusion, using the
  reasoning that is actually stated in the chunks.
- Applying a rule, condition, or threshold from one part of the chunks to
  the facts in another part.
- Comparing two positions, clauses, or holdings that both appear in the
  allowed chunks.
- Combining two facts that are stated separately into a single conclusion.

Avoid questions that mainly ask for:

- a date, registration number, section number, or case citation by itself;
- a named entity or amount that is plainly stated in one sentence;
- a list of items copied verbatim from the text.

Grounding rule: the answer must be directly supported by the allowed
chunks. If the document does not state the reasoning for a "why" question,
do not invent it; choose a different gap or write zero rows. Do not ask
hypothetical "what if" questions, because the chunks cannot contain a
gold answer for a counterfactual.

## What To Look For

Read the visible related training-row bundle and any same-run previous questions first, then inspect the allowed chunks as needed.
Before writing, compare against the related training rows, identify the training intents already covered, and find gaps in question reasoning.
If the related training questions appear to cover the obvious good angles, write fewer rows or zero rows instead of guessing a cosmetic rewrite.
Your target is gaps in question reasoning: what operation, distinction, condition,
exception, relationship, or trap was not tested by the training questions even
though the allowed chunks support it. If same-run previous questions already used a gap,
choose a different gap or write zero rows.

Before writing rows, decide whether the candidate cluster is coherent:

- `coherent`: the chunks form a sensible eval unit.
- `partially coherent`: only a subset of chunks should be used.
- `weak/noisy`: the chunks do not support a good eval question; write zero rows.

Find useful gaps such as:

- an unused concrete fact,
- a reverse lookup from consequence, entity, date, amount, or outcome back to the rule, action, or document,
- an application of a rule or condition to the facts stated in the chunks,
- a combination of two or more facts that were only tested separately,
- an answerable rule or condition,
- an exception or threshold,
- a date or entity distinction,
- a relationship between included chunks,
- a fair negative/trap question directly refuted by the chunks,
- an easier recall question when training asked only a complex one,
- a harder synthesis question when training asked only a narrow extraction.

Do not ask about facts absent from the allowed chunks. Do not assume missing
context. If the packet is weak, noisy, or fragmentary, write fewer rows or zero rows.

## Output

Append JSON objects to the assignment output path. Each JSON object must contain
exactly these CSV fields:

```json
{
  "question": "...",
  "answer": "...",
  "docIds": ["clf-..."],
  "chunk_ids": ["clf-...#12"],
  "pipeline": "Eval-Question-Gen"
}
```

At the end, write the summary path with only:

- reasoning gaps used,
- packet status: `saturated`, `weak`, or `usable`,
- optional one-line notes for issues such as a visible row-count mismatch.
