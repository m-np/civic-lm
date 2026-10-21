# Spec: superseded (v0.1)

**Goal.** Notice when the supplied text is not the rule in force on the asked date, and
answer from the current version when it is supplied.

## Inputs
- `diffs.parquet` rows with `change_type` in {amended, removed, renumbered}. `editorial` and
  `added` rows are never used.
- `section_history.parquet` for the text of the old and new versions.
- Test items use only diffs with `post_cutoff == True` (time split) that are in test parts
  (structure split). Train and dev items use diffs of their own parts from any date.

## Base question
The question targets a **changed paragraph**, one of the `changed_paragraphs` entries with
`change == "modified"`. It is written by the same mechanism as grounded_qa (template or LLM),
from the **new** version of that paragraph. A question is kept only if its gold answer differs
between the two versions: the old version's corresponding paragraph does not contain the new
`answer_span`. Without this check the item would test nothing.

For `removed` diffs: the question asks about the obligation in the removed section. The
correct current answer is that the provision no longer applies.

## Conditions
- `outdated_only`: only the old version is supplied, labeled with its effective date. The
  question asks about the rule "as of {as_of}". Correct: `outdated_warning = true`, plus either
  `abstain = true` or an answer that is explicitly qualified as from the older text. Gold
  `should_abstain = true`.
- `outdated_plus_current`: both versions are supplied with their effective dates, in shuffled
  order. Correct: answer from the current version and cite it. Gold `answer` = the new answer;
  `citations` = the current section.

Control: every `outdated_plus_current` item has a sibling grounded_qa item (current version
only), so "always warn" is penalized.

## How the model can tell
Every passage in every task is rendered with the period it was in force, for example
"in force from 2024-10-15 until 2026-02-04" or "in force from 2020-06-22, no end date recorded"
(from `passage_validity.json`). The question names the date it asks about. Detecting
supersession is then objective: compare the dates. Current passages in other tasks carry the
same kind of header, and `false_outdated_warning_on_current` is the control against a model
that always warns.

## Pairs (as built in v0.1)
For each lineage, the old version is the one in force on `test_cutoff` (test parts) or
`history_start` (train and dev parts), and the new version is the one in force on `as_of`.
Questions currently come from definition changes: a term that was redefined, or a term that
was dropped.

## Gold
`superseded` = `SupersededLabel` (diff_id, from/to section ids and dates, changed paragraph
paths, change_type).

## Metrics
- Accuracy of `outdated_warning` on `outdated_only` and its sibling controls.
- Answer correctness and citation of the current version on `outdated_plus_current`.

## Scope note for v1
Post-cutoff test diffs are sparse. In the 2026-10-20 scope, most of them are in 21 CFR 820
(QMSR, 2026-02-02), 29 CFR 541 (2026-05-15), 21 CFR 312/314, and 40 CFR 60/63. Expect about
60–150 test items. Report this count in article 1.
