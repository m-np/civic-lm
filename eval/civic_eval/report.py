"""Compare scored runs of one release in a markdown table (the headroom view).

    python -m civic_eval.report --release sample          # -> eval/runs/sample/REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .score import score_run

REPO_ROOT = Path(__file__).resolve().parents[2]


def fmt(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def build(release: str) -> str:
    runs = sorted(p for p in (REPO_ROOT / "eval" / "runs" / release).iterdir()
                  if (p / "outputs.jsonl").exists())
    results = [score_run(r) for r in runs]
    keys = sorted({k for r in results for k in r["by_task_condition"]})
    names = [r["run"] for r in results]
    lines = [f"# Results: release `{release}`", "",
             "Accuracy per task and passage condition (n = items). Rules for `correct` are in "
             "`civic_eval/score.py`.", "",
             "| task / condition | n | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names)]
    for k in keys:
        n = next(r["by_task_condition"][k]["n"] for r in results if k in r["by_task_condition"])
        cells = [fmt(r["by_task_condition"].get(k, {}).get("accuracy")) for r in results]
        lines.append(f"| {k} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "| overall metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m in ("format_ok", "abstain_precision", "abstain_recall", "false_outdated_warning_on_current"):
        lines.append(f"| {m} | " + " | ".join(fmt(r["overall"][m]) for r in results) + " |")
    for m in ("answer_f1", "cit_p", "cit_r"):
        k = "grounded_qa/gold_distractors"
        lines.append(f"| {m} (QA + distractors) | " + " | ".join(
            fmt(r["by_task_condition"].get(k, {}).get(m)) for r in results) + " |")
    k = "closed_book/closed_book"
    lines.append("| hallucinated citations (closed-book) | " + " | ".join(
        fmt(r["by_task_condition"].get(k, {}).get("hallucinated")) for r in results) + " |")
    lines.append("| mean latency (s/item) | " + " | ".join(fmt(r["mean_latency_s"]) for r in results) + " |")
    lines += ["", "Runs:"] + [f"- `{r['run']}`: {r['model']} @ {str(r['revision'])[:10]}, tier {r['tier']}"
                              for r in results]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="sample")
    args = ap.parse_args(argv)
    md = build(args.release)
    out = REPO_ROOT / "eval" / "runs" / args.release / "REPORT.md"
    out.write_text(md)
    print(md)
    print(json.dumps({"written": str(out)}))


if __name__ == "__main__":
    main()
