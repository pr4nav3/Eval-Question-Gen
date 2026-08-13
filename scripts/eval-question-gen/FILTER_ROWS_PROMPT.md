# Training Row Filter

You are filtering rows from a SEBI question-answer dataset.

## Goal of the dataset

Train a model to remember the SEBI document corpus: facts, rules, dates, actors, exceptions, procedures, document lineages, and practical relationships across as much of the corpus as possible.

Hard, evidence-grounded, single-document or multi-document QAs are the desired training signal. Coverage and memory density matter more than ornamental novelty. Uniqueness only prevents duplicate or same-intent rows that waste training budget.

## Your task

Read the provided `Question` and `Answer`. Decide whether to **KEEP** or **REJECT** the row.

Return exactly this JSON:

```json
{
  "decision": "KEEP",
  "reason": "coherent QA that teaches a SEBI memory target"
}
```

or

```json
{
  "decision": "REJECT",
  "reason": "mentions cited chunks"
}
```

The `reason` should be one short phrase (e.g., "template + evidence container", "very short answer", "incoherent answer", "exact duplicate").

## Hard reject — remove these immediately

### 1. Templated questions that mention evidence containers
Reject fixed-template questions that name the source material rather than the underlying fact. Examples include:

- "Compare the SEBI documents '...' and '...' ..."
- "Use the cited chunks from '...' and '...' ..."
- "Using the SEBI document '...', reconstruct ..."
- "A model says the SEBI document '...' can be answered from one opening paragraph..."
- "A practitioner reads only the opening of '...' ..."

These are bad because they teach the model to reason about documents and chunks, not about SEBI facts.

### 2. Evidence-container language
Reject questions that refer to the packaging of the evidence, even if not a fixed template:

- "cited chunks", "chunk", "chunks"
- "SEBI document", "the document", "this document", "documents"
- "table", "Table VII", "five-row table"
- "opening paragraph", "paragraph", "early and later sections"
- "section", "schedule", "annexure", "appendix" when used as a locational pointer

Regulatory references like "Regulation 6(1)", "Section 15HA", "Paragraph 10 of Schedule B" are fine — they name specific rules, not the evidence container.

### 3. Exact duplicates
If the question and answer are substantively identical to an earlier row, reject it as a duplicate.

### 4. Very short or logically incoherent answers
Reject rows where:

- The answer is just a bare fragment (e.g., "MSEI; Table 50", "USDJPY, with 847 lots") with no explanatory context.
- The answer is too short to meaningfully address the question.
- The answer is grammatically or logically nonsensical given the question.
- The answer appears to be a list item or table cell rather than a real answer.

## Selective reject — use judgment

Some template-style questions do **not** mention evidence containers. Evaluate these individually:

- "Reconstruct the chronological or procedural relationship between ..."
- "A participant relies on ... to answer a question governed by ..."
- "Reconcile the key figures, dates, or quantities across ..."
- "Reconstruct the chain from factual evidence through legal analysis ..."
- "Using only the SCRA Amendment Rules, compare ..."

Keep these if they ask a coherent, memory-target question and the answer teaches a real SEBI fact. Reject them if the answer is thin, generic, or feels like rote filler.

## Do NOT reject

- Questions that name SEBI instruments (recovery certificate, adjudication order, notice of demand, notice of attachment, informal guidance, consultation paper, SEBI order for compliance, etc.). These are memory targets and are fine.
- Long questions. Length is fine if the question is coherent and the answer is substantive.
- Single-document or multi-document questions. Both are valid.
- SEBI jargon or regulation names. This is expected.
- Answers that cite specific regulations, dates, amounts, or actors, as long as they make sense.

## Examples

KEEP:
Q: What interpretation did SEBI provide regarding the term 'level' in the Proviso to Regulation 6(1) of the LODR Regulations when responding to DCB Bank's query about Ms. Rubi Chaturvedi's position?
A: SEBI interpreted that the term 'level' refers to the position a person occupies in the organizational hierarchy, distinct from 'reporting'...
Reasoning: Names a specific SEBI rule and actor; answer is substantive.

KEEP:
Q: Summarize SEBI's view across the Sandhar, PI Industries and MPS family trust exemption orders.
A: SEBI treated all three as promoter-family restructurings rather than substantive changes in public ownership or control...
Reasoning: Multi-document synthesis that teaches document lineage and SEBI reasoning.

REJECT:
Q: Use the cited chunks from 'Sale Confirmation Order...' and 'Warning Letter...' to answer a cross-document applicability question...
Reasoning: Mentions "cited chunks" and is a fixed template.

REJECT:
Q: A model says the SEBI document 'Notice Of Attachment...' can be answered from one opening paragraph alone. Is that safe?
Reasoning: Template + mentions "SEBI document" and "opening paragraph".

REJECT:
Q: In Table 51 of the SEBI Bulletin, BSE reported zero Interest Rate Futures trading volumes...
A: 2,012 and 2,269, indicated by the $ symbol
Reasoning: Answer is a bare fragment with no explanatory context.

REJECT:
Q: For the Orchid-Dhanuka scheme, what meetings were called and on what schedule?
A: <exact same wording repeated from an earlier row>
Reasoning: Exact duplicate of an earlier row.

## Important

Be conservative. If a row is borderline but the question is coherent and the answer looks meaningful, KEEP it. Only reject obviously bad rows. Use your own judgment for cases not covered by the examples above.
