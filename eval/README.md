# eval/

Harness, scoring and judge (Oct 27 – Nov 2). Reads only the frozen release in `bench/releases/`
and `civic_bench.schema`. Never imports `civic_data` (enforced by `lint-imports`).

Planned modules: `runner.py` (vLLM + API clients, temperature 0, fixed seeds, full logging to
`eval/runs/`), `prompts.py` (naive zero-shot and engineered few-shot tiers), `score/`
(citation, abstain, superseded, rouge, support), `judge.py` + `calibrate.py` (judge model must
differ from the question-writing model; calibrate on about 150 human labels), `report.py`
(headroom table).
