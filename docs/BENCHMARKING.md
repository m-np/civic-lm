# How to benchmark a model with civic-lm

This guide runs the whole path, from raw regulations to a results table. It also explains
what each number means. Only open-weight models are used, because the same models are
fine-tuned later.

```
eCFR / Federal Register ──ingest──▶ data/raw ──normalize/versions──▶ data/processed
      ──civic_bench.build──▶ bench/releases/<name>/items.jsonl
      ──civic_eval.run──▶ eval/runs/<name>/<model>__<tier>/outputs.jsonl
      ──civic_eval.score / report──▶ scores.json, REPORT.md
```

## 0. Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[data,dev,eval]"     # add ".[vllm]" on a GPU machine
```

If `huggingface_hub` picks up an expired saved token, public models fail to download. Set
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, or run `hf auth login` again. The gated Llama models need
a valid token and an accepted license.

## 1. Data (stages 1–3)

```bash
# sample: two parts. Leave out --only for the full scope in configs/scope.yaml.
.venv/bin/python -m civic_data.ingest.ecfr   --only 40:122 21:820
.venv/bin/python -m civic_data.ingest.fedreg --only 40:122 21:820
make data          # sections, provenance, section_history, diffs
make spotcheck     # gate: hand-check 50 diffs in data/interim/diff_spotcheck.csv
```

## 2. Build a release (stages 4–5)

```bash
.venv/bin/python -m civic_bench.build --name sample --n-definitions 12 --n-superseded 8
```

This writes `bench/releases/sample/`:

| File | What it is |
|---|---|
| `items.jsonl` | One `civic_bench.schema.Item` per line: question, passages, gold, split, provenance |
| `MANIFEST.json` | Counts per task, condition and split, plus the sha256 of `items.jsonl` (every run records it) |
| `passage_validity.json` | When each passage version was in force (shown to the model) |
| `valid_sections.json` | Every real section, used to detect invented citations |

A release is frozen once it is built. To change items, build a new release with a new name,
so that results from different releases are never mixed.

## 3. Run a model

Models are listed in `configs/models.yaml`, each with a backend and a pinned revision:

| Backend | When to use it |
|---|---|
| `hf` | transformers. Runs on CPU (slow, small models only) or on one GPU |
| `vllm` | GPU machines. This is the main harness for the 7–8B and larger models |
| `ollama` | Local quantized GGUF on CPU. A rough baseline only; not comparable to bf16 runs |

```bash
.venv/bin/python -m civic_eval.run --release bench/releases/sample --model qwen2.5-1.5b --tier naive
.venv/bin/python -m civic_eval.run --release bench/releases/sample --model qwen2.5-1.5b --tier engineered
# or: make run MODEL=qwen2.5-7b TIER=engineered
```

Every run decodes greedily (temperature 0) with a fixed seed. It logs each full prompt, the
raw output, token counts and latency to `eval/runs/<release>/<model>__<tier>/outputs.jsonl`.
`run.json` records the model revision, the prompt version and the release hash. Runs resume:
if a run is interrupted, re-run the same command.

**Prompt tiers.** Every tier gets identical passages; only the instructions differ.
- `naive`: one instruction plus the JSON format.
- `engineered`: rules for citing, abstaining and checking whether a passage was in force,
  plus 3 few-shot examples. The examples are written about a fictional "Title 99", so they
  give the model no benchmark content.

The fine-tuned model must beat the base model's *engineered* tier, not just the naive one.

To add a model, add an entry to `configs/models.yaml` with `backend`, `model` (the HF repo id)
and `revision` (a commit sha).

## 4. Score and compare

```bash
.venv/bin/python -m civic_eval.score eval/runs/sample/qwen2.5-1.5b__engineered
.venv/bin/python -m civic_eval.report --release sample     # -> eval/runs/sample/REPORT.md
```

### What is measured

| Task / condition | What the model sees | What counts as correct |
|---|---|---|
| `grounded_qa/gold_only` | the gold passage | answer token-F1 ≥ 0.5 with the gold **and** a gold section cited, without abstaining |
| `grounded_qa/gold_distractors` | the gold passage plus 3 lexically similar sections | same |
| `abstention/no_gold` | nearby passages that do not contain the answer | `abstain: true` |
| `superseded/outdated_only` | only a version that is no longer in force (its dates are shown) | flags it (`outdated_warning` or `abstain`) |
| `superseded/outdated_plus_current` | the old and current versions, dated | answers from the current version; for a definition that was dropped, flags it |
| `closed_book/closed_book` | no passages | same as QA. This is a control: watch the **hallucinated citation** rate, not accuracy |

Overall metrics:
- **abstain precision and recall:** computed over all open-book items. A model that always
  abstains gets high recall and fails grounded_qa.
- **false_outdated_warning_on_current:** a warning raised on current passages. This is the
  control for "always warn".
- **cit_p / cit_r:** citation precision and recall at section level. Citing a paragraph inside
  the gold section counts as correct.
- **hallucinated:** the share of cited sections that do not exist. Only sections in ingested
  parts can be checked.

The answer metric is currently token-F1, which can be fooled by paraphrase. The plan's LLM
judge, built from an open model different from any question writer and calibrated on about
150 human labels, will replace it for answer quality.

### Reading the results: the headroom gate

The report is the headroom check from the plan. For each task:
- **Saturated:** the strongest baseline scores above about 95%. The task cannot show a gain
  from tuning. Make it harder or drop it.
- **Broken:** every model scores near 0. Check the items by hand before blaming the models.
- **Good:** a clear spread, for example small models at 40–60% and the large open model at
  80–90%.

Compare the train and test splits as well (`scored.jsonl` has `split` on every row). Test
items come from held-out parts and from post-cutoff amendments.

## 5. Later: the fine-tuned model

The tuned model runs through exactly the same command, as a new `configs/models.yaml` entry
that points at the merged checkpoint or at a local path. The report then shows base and tuned
side by side on the same frozen release.
