# civic-lm

A benchmark, and later a fine-tuned model, for **grounded government Q&A**: answer from supplied
regulatory text, cite the right CFR section, abstain when the text does not answer, and notice
when the text has been superseded.

Labels are made **by construction** from the structure of the eCFR, not by an LLM or a human
where possible. A question is built from one section, so the gold citation is known. Dated
section versions give exact "amended / removed / renumbered" labels.

## Layout

| Folder | Stage | Reads | Writes |
|---|---|---|---|
| `data/` (`civic_data`) | 1 ingest · 2 normalize · 3 version store | APIs, `configs/scope.yaml` | `data/raw/` (append-only), `data/processed/*.parquet` |
| `bench/` (`civic_bench`) | 4 generators · 5 validate/freeze | `data/processed/` | `bench/releases/` |
| `eval/` (`civic_eval`) | harness, scoring, judge | `bench/releases/` | `eval/runs/` |
| `train/` (`civic_train`) | SFT + LoRA | the bench train split | adapters |

Dependency rule: `data → bench → eval`, and `train` reads only `bench`. The folders share data
only through files. `civic_bench.schema` is the one shared code contract. `lint-imports`
enforces the rule in CI.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[data,dev]"
make test
make ingest                 # or: python -m civic_data.ingest.ecfr --only 40:122 21:820
make data                   # sections, provenance, section_history, diffs
make spotcheck              # gate: 50 diffs for hand review -> data/interim/diff_spotcheck.csv
make lock && make rebuild   # gate: a rebuild from raw/ reproduces the recorded hashes
```

## Processed tables

- `sections.parquet`: one row per (section, snapshot date) for history_start, test_cutoff
  and as_of. Holds labeled paragraphs (`(b)(14)(ii)`), defined terms, the FR citation note,
  and a text hash.
- `provenance.parquet`: every raw file, with its source URL, license, retrieval time and sha256.
- `section_history.parquet`: every section version in (history_start, as_of].
- `diffs.parquet`: `added | amended | removed | renumbered | editorial`, with changed
  paragraphs, similarity, amending Federal Register documents, split and `post_cutoff`.

Each section is checked against the raw XML when it is parsed: the normalized text must equal
the raw text, with whitespace removed.

## Splits

- **By structure:** whole CFR parts are assigned to train, dev or test in
  `configs/scope.yaml`.
- **By time:** test items for superseded detection come from amendments effective on or after
  `test_cutoff`.

The test parts were chosen because they have post-cutoff amendments.

## Benchmarking

See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for the full guide: build a release, run
open-weight models (transformers, vLLM or Ollama), and score and compare the results.

```bash
make bench RELEASE=sample
make run   RELEASE=sample MODEL=qwen2.5-1.5b TIER=engineered
make report RELEASE=sample
```

## Status (2026-10-20)

- Stages 1–3 run on real data.
- Template generators (definition QA, abstention, closed-book, superseded) and the eval
  harness (two prompt tiers, automatic scoring, report) work end to end on a 50-item sample.
- Next: LLM-written questions, validation and dedupe checks, the LLM judge, the full ingest
  and the 200-item pilot.
