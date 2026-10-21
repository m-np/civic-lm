# Spec: obligation (v0.1, stretch: build only if the pilot gate passes on time)

**Goal.** Extract who must do what, under which condition, and by when.

## Construction
1. Candidate paragraphs: labeled paragraphs with a modal verb (`shall`, `must`, `may not`,
   `is required to`) and 15–200 words, from test and train parts.
2. Template extraction pre-labels them. A rule-based pass splits on the modal verb:
   actor = the noun phrase before it, action = the verb phrase after it, deadline = a
   `within|no later than|by` phrase, condition = an `if|when|unless` clause.
3. **Human check:** every test item is corrected by a reviewer in the review sheet. Only
   items marked `human_verified = true` enter the test split.

## Gold
`obligations: list[Obligation]` (actor, action, modality, condition, deadline, citation).

## Metric
Field-level F1. Actor and action are matched with token-F1 ≥ 0.6; modality is matched
exactly. Deadline and condition are matched with token-F1 ≥ 0.6 when they are present in the gold.

## Why stretch
The labels need human correction for every item. That effort competes with the 200-item
verification budget before Nov 2.
