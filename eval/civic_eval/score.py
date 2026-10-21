"""Automatic scoring of a run (no LLM judge yet: answer quality uses token-F1 against the gold).

    python -m civic_eval.score eval/runs/sample/qwen2.5-1.5b__engineered

Writes scored.jsonl (per item) and scores.json (per task/condition and overall) in the run dir.

Per item:
  format_ok        output parsed as the required JSON object
  answer_f1        token-F1 of answer vs gold answer (answerable items)
  cit_p / cit_r    section-level citation precision / recall (a paragraph inside the gold
                   section counts; acceptable_citations count for precision)
  correct          answerable: not abstained, answer_f1 >= 0.5 and cites a gold section
                   abstention: abstained
                   superseded: flagged (outdated_warning or abstain); for a redefined term with
                   both versions given, also answered from the current version
  hallucinated     share of cited sections that do not exist in the source tables
                   (only sections in ingested parts can be checked; others are "unverifiable")
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from civic_bench.schema import Condition, Item, ModelAnswer, Task

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of to in on for by or and is are be any as at with from that this which such
under shall may must means mean its it their than other not no all each""".split())
_CIT = re.compile(r"(\d+)\s*C\.?F\.?R\.?\s*(?:§+\s*)?(\d+[A-Za-z]?\.\d+[A-Za-z0-9\-]*)((?:\([A-Za-z0-9]+\))*)")
F1_CORRECT = 0.5


def tokens(s: str) -> list[str]:
    return [t for t in _TOKEN.findall((s or "").lower()) if t not in _STOP]


def token_f1(pred: str | None, gold: str | None) -> float:
    p, g = Counter(tokens(pred or "")), Counter(tokens(gold or ""))
    common = sum((p & g).values())
    if not common:
        return 0.0
    pr, rc = common / sum(p.values()), common / sum(g.values())
    return 2 * pr * rc / (pr + rc)


def parse_output(text: str) -> ModelAnswer | None:
    """First JSON object in the output; tolerant of code fences and surrounding prose."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[start:i + 1])
                        if isinstance(d.get("citations"), str):
                            d["citations"] = [d["citations"]]
                        d["citations"] = [str(c) for c in d.get("citations") or []]
                        if d.get("answer") is not None and not isinstance(d["answer"], str):
                            d["answer"] = json.dumps(d["answer"])
                        return ModelAnswer.model_validate(d)
                    except (ValueError, TypeError):
                        break
        start = text.find("{", start + 1)
    return None


def sections_of(citations: list[str]) -> list[str]:
    """Normalize citations to section level: '40 CFR § 122.26(b)(14)' -> '40 CFR 122.26'."""
    out = []
    for c in citations:
        for m in _CIT.finditer(c):
            out.append(f"{m.group(1)} CFR {m.group(2)}")
    return list(dict.fromkeys(out))


def score_item(item: Item, out: ModelAnswer | None, valid: set[str], scoped_parts: set[str]) -> dict:
    g = item.gold
    r = {"item_id": item.item_id, "task": item.task.value, "condition": item.condition.value,
         "split": item.split, "format_ok": out is not None}
    out = out or ModelAnswer()
    pred_secs = sections_of(out.citations)
    gold_secs = set(sections_of(g.citations))
    ok_secs = gold_secs | set(sections_of(g.acceptable_citations))
    r.update(abstain=out.abstain, warn=out.outdated_warning, gold_abstain=g.should_abstain,
             n_cited=len(pred_secs))
    if gold_secs:
        hit = [s for s in pred_secs if s in ok_secs]
        r["cit_p"] = len(hit) / len(pred_secs) if pred_secs else 0.0
        r["cit_r"] = len(gold_secs & set(pred_secs)) / len(gold_secs)
    if g.answer:
        r["answer_f1"] = round(token_f1(out.answer, g.answer), 4)
    checkable = [s for s in pred_secs if s.split(".")[0].replace(" CFR ", ":") in scoped_parts]
    if checkable:
        r["hallucinated"] = sum(s not in valid for s in checkable) / len(checkable)

    if item.task in (Task.grounded_qa, Task.closed_book):
        r["correct"] = (not out.abstain and r.get("answer_f1", 0) >= F1_CORRECT and r.get("cit_r", 0) > 0)
    elif item.task == Task.abstention:
        r["correct"] = out.abstain
    elif item.task == Task.superseded:
        flagged = out.outdated_warning or out.abstain
        if item.condition == Condition.outdated_plus_current and g.answer:
            f_new = token_f1(out.answer, g.answer)
            f_old = max([token_f1(out.answer, a) for a in g.answer_aliases] or [0.0])
            r["used_current"] = f_new > f_old and f_new >= F1_CORRECT
            r["correct"] = r["used_current"]
        else:
            r["correct"] = flagged
    return r


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['task']}/{r['condition']}"].append(r)
    by = {}
    for k, rs in sorted(groups.items()):
        by[k] = {"n": len(rs), "accuracy": _mean([float(r["correct"]) for r in rs]),
                 "format_ok": _mean([float(r["format_ok"]) for r in rs]),
                 "answer_f1": _mean([r.get("answer_f1") for r in rs]),
                 "cit_p": _mean([r.get("cit_p") for r in rs]), "cit_r": _mean([r.get("cit_r") for r in rs]),
                 "abstain_rate": _mean([float(r["abstain"]) for r in rs]),
                 "warn_rate": _mean([float(r["warn"]) for r in rs]),
                 "hallucinated": _mean([r.get("hallucinated") for r in rs])}
    # abstention precision/recall over every open-book item
    ob = [r for r in rows if r["task"] != "closed_book"]
    tp = sum(r["abstain"] and r["gold_abstain"] for r in ob)
    pred, gold = sum(r["abstain"] for r in ob), sum(r["gold_abstain"] for r in ob)
    answerable = [r for r in rows if r["task"] == "grounded_qa"]
    overall = {
        "n": len(rows),
        "format_ok": _mean([float(r["format_ok"]) for r in rows]),
        "abstain_precision": round(tp / pred, 4) if pred else None,
        "abstain_recall": round(tp / gold, 4) if gold else None,
        "false_outdated_warning_on_current": _mean([float(r["warn"]) for r in answerable]),
        "task_accuracy": {t: _mean([float(r["correct"]) for r in rows if r["task"] == t])
                          for t in sorted({r["task"] for r in rows})},
    }
    return {"overall": overall, "by_task_condition": by}


def score_run(run: Path, release: Path | None = None) -> dict:
    meta = json.loads((run / "run.json").read_text())
    release = release or Path(__file__).resolve().parents[2] / "bench" / "releases" / meta["release"]
    items = {i.item_id: i for i in (Item.model_validate_json(x) for x in open(release / "items.jsonl"))}
    valid = set(json.loads((release / "valid_sections.json").read_text()))
    scoped = {s.split(".")[0].replace(" CFR ", ":") for s in valid}
    from .run import load_outputs

    rows = []
    for o in load_outputs(run):
        rows.append({**score_item(items[o["item_id"]], parse_output(o["output"]), valid, scoped),
                     "latency_s": o["latency_s"]})
    with open(run / "scored.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"run": run.name, "model": meta["model"], "revision": meta.get("revision"), "tier": meta["tier"],
           "release": meta["release"], "mean_latency_s": _mean([r["latency_s"] for r in rows]), **aggregate(rows)}
    (run / "scores.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args(argv)
    for run in args.runs:
        res = score_run(run)
        print(json.dumps({"run": res["run"], **res["overall"]}, indent=2))


if __name__ == "__main__":
    main()
