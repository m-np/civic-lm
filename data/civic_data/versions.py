"""Stage 3: version store -> data/processed/section_history.parquet + diffs.parquet.

section_history: one row per (lineage, version date) from history_start to as_of.
  Sources: the whole-part snapshot at history_start (baseline), the per-version section
  files fetched by ingest, and "removed" markers from the versioner API.
diffs: one row per change between consecutive versions (see diff_classify).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, Scope, load_scope
from .diff_classify import Version, classify_lineage, diffs_to_rows, pair_renumberings
from .ids import lineage_id, section_id
from .ingest.ecfr import section_rel, section_versions, versions_rel
from .normalize import VersionIndex, parse_file

log = logging.getLogger(__name__)

HISTORY_COLS = ["lineage_id", "section_id", "title", "part", "split", "section", "version_date",
                "substantive", "removed", "reserved", "heading", "text", "text_hash",
                "paragraphs", "source_path"]


def _load_fedreg(raw: Path, title: str, part: str) -> list[dict]:
    docs = []
    for page in sorted((raw / f"fedreg/title-{title}/part-{part}").glob("page-*.json")):
        docs.extend(json.loads(page.read_text()).get("results", []))
    return docs


def part_history(scope: Scope, sections: pd.DataFrame, title: str, part: str, split: str) -> list[dict]:
    raw = scope.paths.raw
    vjson = json.loads((raw / versions_rel(title, part)).read_text())
    index = VersionIndex(vjson)
    meta = {(v["identifier"], v["date"]): v for v in section_versions(vjson)}
    rows: dict[str, dict] = {}

    def add(rec: dict) -> None:
        prev = rows.get(rec["section_id"])
        if prev and prev["text_hash"] != rec["text_hash"] and not rec["removed"]:
            log.warning("two texts for %s (%s vs %s); keeping the per-version file",
                        rec["section_id"], prev["source_path"], rec["source_path"])
            if "/section-" not in rec["source_path"]:
                return
        rows[rec["section_id"]] = rec

    base = sections[(sections.title == title) & (sections.part == part)
                    & (sections.snapshot_date == scope.history_start)]
    for r in base.to_dict("records"):
        add(_row(r, split, meta.get((r["section"], r["version_date"]))))

    for v in section_versions(vjson):
        if not (scope.history_start < v["date"] <= scope.as_of):
            continue
        if v["removed"]:
            add({"lineage_id": lineage_id(title, v["identifier"]),
                 "section_id": section_id(title, v["identifier"], v["date"]),
                 "title": title, "part": part, "split": split, "section": v["identifier"],
                 "version_date": v["date"], "substantive": v["substantive"], "removed": True,
                 "reserved": False, "heading": v["name"], "text": "", "text_hash": "",
                 "paragraphs": "[]", "source_path": versions_rel(title, part)})
            continue
        if not v["substantive"]:
            continue
        p = raw / section_rel(title, part, v["identifier"], v["date"])
        if not p.exists():
            continue
        for rec in parse_file(p, raw_dir=raw, title=title, snapshot_date=v["date"], index=index):
            d = {"lineage_id": rec.lineage_id, "section_id": rec.section_id, "section": rec.section,
                 "version_date": rec.version_date, "reserved": rec.reserved, "heading": rec.heading,
                 "text": rec.text, "text_hash": rec.text_hash, "source_path": rec.source_path,
                 "paragraphs": json.dumps([vars(x) for x in rec.paragraphs], ensure_ascii=False)}
            add(_row(d, split, meta.get((rec.section, rec.version_date)), title=title, part=part))
    return list(rows.values())


def _row(r: dict, split: str, meta: dict | None, title: str | None = None, part: str | None = None) -> dict:
    return {
        "lineage_id": r["lineage_id"], "section_id": r["section_id"],
        "title": title or r["title"], "part": part or r["part"], "split": split,
        "section": r["section"], "version_date": r["version_date"],
        "substantive": bool(meta["substantive"]) if meta else True,
        "removed": bool(meta["removed"]) if meta else False,
        "reserved": bool(r["reserved"]), "heading": r["heading"], "text": r["text"],
        "text_hash": r["text_hash"], "paragraphs": r["paragraphs"], "source_path": r["source_path"],
    }


def build(scope: Scope) -> tuple[pd.DataFrame, pd.DataFrame]:
    sections = pd.read_parquet(scope.paths.processed / "sections.parquet")
    history_rows: list[dict] = []
    all_diffs = []
    for title, part, split in scope.parts():
        if not (scope.paths.raw / versions_rel(title, part)).exists():
            continue
        rows = part_history(scope, sections, title, part, split)
        history_rows.extend(rows)
        vjson = json.loads((scope.paths.raw / versions_rel(title, part)).read_text())
        first_date = min((v["date"] for v in section_versions(vjson)), default=None)
        by_lineage: dict[str, list[Version]] = defaultdict(list)
        for r in rows:
            by_lineage[r["lineage_id"]].append(Version(
                lineage_id=r["lineage_id"], section_id=r["section_id"], title=title, part=part,
                section=r["section"], version_date=r["version_date"], text=r["text"],
                text_hash=r["text_hash"], paragraphs=json.loads(r["paragraphs"]),
                substantive=r["substantive"], removed=r["removed"], reserved=r["reserved"]))
        part_diffs = []
        for lid in sorted(by_lineage):
            vs = sorted(by_lineage[lid], key=lambda v: v.version_date)
            part_diffs.extend(classify_lineage(vs, history_start=scope.history_start,
                                               part_first_date=first_date))
        text_of = {r["section_id"]: r["text"] for r in rows}
        part_diffs = pair_renumberings(part_diffs, text_of)
        fr = _load_fedreg(scope.paths.raw, title, part)
        for d in part_diffs:
            d.fr_documents = sorted({doc["document_number"] for doc in fr
                                     if d.effective_date in (doc.get("effective_on"), doc.get("publication_date"))})
        all_diffs.extend(part_diffs)
        log.info("%s CFR %s: %d versions, %d diffs", title, part, len(rows), len(part_diffs))

    hist = pd.DataFrame(history_rows, columns=HISTORY_COLS)
    hist = hist.sort_values(["title", "part", "lineage_id", "version_date"], kind="stable").reset_index(drop=True)
    diffs = pd.DataFrame(diffs_to_rows(all_diffs))
    if not diffs.empty:
        diffs["split"] = [scope.split_of(t, p) for t, p in zip(diffs["title"], diffs["part"], strict=False)]
        diffs["post_cutoff"] = diffs["effective_date"] >= scope.test_cutoff
        for col in ("changed_paragraphs", "fr_documents"):
            diffs[col] = diffs[col].map(lambda x: json.dumps(x, ensure_ascii=False))
    return hist, diffs


def main(argv: list[str] | None = None) -> None:
    from .manifest import write_tables

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scope = load_scope(args.config, args.data_dir)
    hist, diffs = build(scope)
    write_tables(scope, {"section_history": hist, "diffs": diffs})
    if not diffs.empty:
        log.info("diff counts:\n%s", diffs.groupby(["split", "change_type"]).size().to_string())


if __name__ == "__main__":
    main()
