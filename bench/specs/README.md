# Generator specifications (v1)

Each task has one generator script in `bench/civic_bench/generators/` and one spec here.
A spec is the contract for its generator. Change the spec first, then the code, then bump
`generator_version`.

| Task | Spec | Gold label from | v1 status |
|---|---|---|---|
| Grounded QA with citation | [grounded_qa.md](grounded_qa.md) | the source section | in |
| Abstention | [abstention.md](abstention.md) | construction | in |
| Superseded detection | [superseded.md](superseded.md) | `diffs.parquet` | in for scoped titles; dropped to v1.1 if the version-store gate slips past Oct 24 |
| Plain-language summary | [summary.md](summary.md) | BillSum / GovReport references | in |
| Closed-book control | [closed_book.md](closed_book.md) | the source section | in (small) |
| Obligation extraction | [obligation.md](obligation.md) | template extraction + human check | stretch |

## Rules for all generators

**Inputs.** Generators read only `data/processed/*.parquet` and `configs/`. They never call
the network, except for the LLM question writer in `grounded_qa`, and its outputs are cached
in `data/interim/question_cache/` keyed by prompt hash. A rebuild reuses the cache, so it is
deterministic.

**Determinism.** Each generator takes `--seed` and uses a single `random.Random(seed)`. The
same seed and the same processed tables give byte-identical JSONL. A test checks this.

**One gold section per base item.** Every base item is built from one lineage (`lineage_id`),
so its gold citation is known by construction. The split is fixed by the part that the
lineage belongs to (`configs/scope.yaml`). Generators never choose the split.

**Phrasing variety.** Every template-based question has at least 4 paraphrase templates
(`template_id` is recorded). No template may produce more than 30% of a task's items. This
guards against models learning the template instead of the task.

**No answer leakage in questions.** A question must not contain the section number, the
paragraph label, or a verbatim span of more than 8 tokens from the gold paragraph. `validate.py`
enforces this.

**Passage rendering.** Passages are shown to the model as:

```
[P1] 40 CFR 122.2 — Definitions. (version effective 2019-02-12)
<text>
```

Passage order is shuffled with the item seed. The gold passage is never always first.

**The `as_of` date.** Every item asks about the rule in force on `as_of` (default
`configs/scope.yaml: as_of`). Superseded items depend on this date, and other items state it
for consistency.

**Model output contract** (`civic_bench.schema.ModelAnswer`). Every prompt asks for JSON:

```json
{"answer": "...", "citations": ["40 CFR 122.2"], "abstain": false, "outdated_warning": false}
```

`abstain=true` means "the supplied passages do not answer this". `outdated_warning=true`
means "the supplied text appears to be superseded as of the asked date".

**Citation normalization.** Gold and predicted citations are normalized with
`civic_data.ids.parse_citation`-equivalent logic in eval: `§` and extra spaces are dropped.
A prediction that cites a paragraph inside the gold section (`40 CFR 122.26(b)(14)` when the
gold is `40 CFR 122.26`) counts as correct. Citing the parent section of a gold paragraph
citation counts as `acceptable_citations` (partial credit; see each spec).

**Distractors** (`conditions.py`). Hard distractors come from the same title and the same
split, picked by BM25 over section text with the question as the query, after removing the
gold lineage and its renumbering partners. Each `gold_distractors` item has 3 distractors.
The distractors used are recorded on the item.

**Volume targets (pilot / v1 test).** Pilot: 200 items across tasks. v1: about 1,000 test
items, of which about 200 are human-verified. Rough mix: QA 40%, abstention 20%, superseded
15%, summary 15%, closed-book 10%.
