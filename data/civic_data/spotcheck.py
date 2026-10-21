"""Gate 2 helper: sample diffs for hand review -> data/interim/diff_spotcheck.csv.

Reviewers fill `reviewer_ok` (y/n) and `notes`. Sampling is stratified by change type
and seeded, so the same sheet comes out on every run.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import DEFAULT_CONFIG, load_scope


def _para_text(paragraphs_json: str | None, paths: set[str]) -> str:
    if not paragraphs_json:
        return ""
    ps = json.loads(paragraphs_json)
    return "\n".join(f"[{p['label_path'] or 'intro'}] {p['text']}" for p in ps if p["label_path"] in paths)[:3000]


def build_sheet(diffs: pd.DataFrame, history: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    usable = diffs[diffs.change_type != "editorial"]
    per_type = max(1, n // max(1, usable.change_type.nunique()))
    sample = pd.concat([g.sample(min(len(g), per_type), random_state=seed)
                        for _, g in usable.groupby("change_type", sort=True)])
    sample = pd.concat([sample, usable.drop(sample.index).sample(
        min(len(usable) - len(sample), max(0, n - len(sample))), random_state=seed)])
    paras = history.set_index("section_id")["paragraphs"].to_dict()
    rows = []
    for d in sample.sort_values(["change_type", "diff_id"]).to_dict("records"):
        changed = json.loads(d["changed_paragraphs"])
        paths = {c["label_path"] for c in changed}
        rows.append({
            "diff_id": d["diff_id"], "change_type": d["change_type"], "effective_date": d["effective_date"],
            "from_section_id": d["from_section_id"], "to_section_id": d["to_section_id"],
            "similarity": d["similarity"], "fr_documents": d["fr_documents"],
            "changed_paragraphs": "; ".join(f"{c['label_path'] or '(intro)'}:{c['change']}" for c in changed),
            "old_text": _para_text(paras.get(_sid(d["from_section_id"])), paths),
            "new_text": _para_text(paras.get(_sid(d["to_section_id"])), paths),
            "ecfr_compare_url": _compare_url(d),
            "reviewer_ok": "", "notes": "",
        })
    return pd.DataFrame(rows)


def _sid(v) -> str | None:
    return v if isinstance(v, str) and v else None


def _compare_url(d: dict) -> str:
    """eCFR's own side-by-side view, to check our diff against."""
    new, old = _sid(d["to_section_id"]), _sid(d["from_section_id"])
    title, rest = (new or old).split("-CFR-")
    section = rest.split("@")[0]
    old_date = old.split("@")[1] if old else d["effective_date"]
    return f"https://www.ecfr.gov/compare/{d['effective_date']}/to/{old_date}/title-{title}/section-{section}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)
    scope = load_scope(args.config, args.data_dir)
    p = scope.paths.processed
    sheet = build_sheet(pd.read_parquet(p / "diffs.parquet"), pd.read_parquet(p / "section_history.parquet"),
                        args.n, args.seed)
    out = scope.paths.interim / "diff_spotcheck.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)
    print(f"wrote {len(sheet)} rows to {out}")


if __name__ == "__main__":
    main()
