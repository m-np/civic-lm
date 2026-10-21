# Spec: abstention (v0.1)

**Goal.** When the supplied passages do not answer the question, say so (`abstain=true`)
and do not guess.

## Construction (the label is known by construction)
Start from a `grounded_qa` base question Q with gold lineage G. Supply passages that look
relevant but do not contain the answer:
1. **Near-miss neighbors:** sections from the same part as G (adjacent section numbers
   first), excluding G and its renumbering partners. These score high on BM25 for Q.
2. **Cross-reference stubs:** for definition questions, a section that only refers to the
   term ("is defined at § 122.23") without defining it.
3. For definition questions (v0.1 build): the same definitions section with the defining
   paragraph removed (the closest near-miss), plus one neighbor section.
4. 1–4 passages per item.

Leakage guard: no supplied passage may contain the gold `answer_span`, or for templates the
definition text. Any candidate whose token-F1 with the gold answer is above 0.5 is dropped.
Hand review then checks a sample, because a neighbor can still answer Q in other words.

## Balance
Abstention items must not be learnable from surface cues:
- Use the same passage count distribution as `gold_distractors`.
- The same base questions appear answerable in grounded_qa. A model that always abstains
  scores 0 on grounded_qa.

## Gold
`should_abstain = true`, `answer = null`, `citations = []`. `lineage_id` = G, so that
splits and dedupe follow the gold section.

## Metrics
Abstain precision and recall, computed over the union of abstention items and the grounded_qa
items that share base questions. Both are reported, never one alone.
