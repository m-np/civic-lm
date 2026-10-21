<!-- Copy of eval/runs/sample/REPORT.md from 2026-10-20, CPU runs. Sample release: 50 items from 40 CFR 122 + 21 CFR 820. -->
# Results: release `sample`

Accuracy per task and passage condition (n = items). Rules for `correct` are in `civic_eval/score.py`.

| task / condition | n | llama3.1-8b-q4__engineered | qwen2.5-1.5b__engineered | qwen2.5-1.5b__naive |
|---|---|---|---|---|
| abstention/no_gold | 12 | 0.75 | 0.08 | 0.17 |
| closed_book/closed_book | 6 | 0.17 | 0.00 | 0.00 |
| grounded_qa/gold_distractors | 12 | 0.92 | 0.58 | 0.58 |
| grounded_qa/gold_only | 12 | 0.83 | 0.83 | 0.67 |
| superseded/outdated_only | 4 | 0.75 | 0.00 | 1.00 |
| superseded/outdated_plus_current | 4 | 0.00 | 0.00 | 1.00 |

| overall metric | llama3.1-8b-q4__engineered | qwen2.5-1.5b__engineered | qwen2.5-1.5b__naive |
|---|---|---|---|
| format_ok | 1.00 | 1.00 | 0.98 |
| abstain_precision | 1.00 | 1.00 | 1.00 |
| abstain_recall | 0.50 | 0.05 | 0.10 |
| false_outdated_warning_on_current | 0.00 | 0.00 | 0.54 |
| answer_f1 (QA + distractors) | 0.93 | 0.76 | 0.74 |
| cit_p (QA + distractors) | 1.00 | 0.62 | 0.58 |
| cit_r (QA + distractors) | 1.00 | 0.75 | 0.75 |
| hallucinated citations (closed-book) | 0.00 | 0.00 | 0.00 |
| mean latency (s/item) | 31.88 | 28.87 | 26.88 |

Runs:
- `llama3.1-8b-q4__engineered`: llama3.1:8b @ None, tier engineered
- `qwen2.5-1.5b__engineered`: Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e, tier engineered
- `qwen2.5-1.5b__naive`: Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e, tier naive

## Reading this sample

- **n is tiny (4–12 per row).** These numbers show the harness works end to end. They are not results.
- **Superseded accuracy must be read with `false_outdated_warning_on_current`.** Qwen naive scores
  1.00 on superseded because it warns on everything (0.54 false warnings on current passages).
- **Both-versions superseded items are hard for all three models.** With old and current versions
  both supplied and dated, no model noticed that the term was dropped from the current rule.
  This task has headroom.
- **Template definition QA nearly saturates for Llama 8B** (0.83–0.92). LLM-written questions
  about requirements are needed for headroom.
- **Abstention separates the models** (0.75 vs 0.08), which is a good spread.
- `llama3.1-8b-q4` is a 4-bit GGUF in Ollama on CPU. It is not comparable to bf16 vLLM runs.
