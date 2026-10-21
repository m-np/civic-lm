# Spec: summary (v0.1)

**Goal.** Write a faithful plain-language summary of a bill or report.

## Inputs
- BillSum (`FiscalNote/billsum`, rev 3d85104, CC0): `text` → `summary`. Use the federal `train`/`test`
  splits. `ca_test` (California bills) is out of scope.
- GovReport (`launch/gov_report`, rev 32feeae, CC-BY-4.0): report → summary. These documents
  are long. Keep reports of 8k tokens or fewer so that the 7–8B models fit them in context.
- Both datasets are public and probably in pretraining data. The task measures summary
  quality, not knowledge, so contamination matters less here. That is stated in article 1.

## Items
- `doc_id` = `billsum:{bill_id}` or `govreport:{id}`. `lineage_id` = `doc_id` (the schema
  allows this for summary). `title`/`part` = `"billsum"`/`"-"` or `"govreport"`/`"-"`.
- Split: BillSum and GovReport test rows go to test, train rows to train. The existing dataset
  splits are kept.
- One passage (`role = "gold"`), the full document text. The prompt asks for at most 150 words.

## Metrics
ROUGE-L against the reference, plus a judge score (faithfulness 1–5, coverage 1–5). The judge
is calibrated on the 150-label human set.
