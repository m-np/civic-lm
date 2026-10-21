# Spec: closed_book (v0.1)

**Goal.** A small control. Does tuning change what the model "knows", and does it invent
citations?

## Items
- About 100 grounded_qa base questions from test parts, sampled with the seed and stratified
  by title, are asked without passages.
- The gold answer and citations are the same as in grounded_qa.

## Metrics
- Answer correctness (judge).
- **Hallucinated-citation rate:** the share of cited sections that do not exist in
  `sections.parquet` (any snapshot) for that title. This is the main number. A tuned model
  that cites non-existent sections more often has learned a bad habit.
- Citation accuracy (secondary). Low scores are expected; closed-book is not a headroom task.
