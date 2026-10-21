"""Stage 1a: eCFR versioner API -> data/raw/ecfr/.

For each part in scope:
  versions/title-{t}/part-{p}.json                 amendment history (every section version)
  full/{date}/title-{t}/part-{p}.xml               whole-part snapshot at each scope snapshot date
  full/{date}/title-{t}/part-{p}/section-{s}.xml   one section at each substantive version date
                                                   in (history_start, as_of]
"""
from __future__ import annotations

import argparse
import json
import logging

from ..config import DEFAULT_CONFIG, Scope, load_scope
from ..http import RawStore

log = logging.getLogger(__name__)


def versions_rel(title: str, part: str) -> str:
    return f"ecfr/versions/title-{title}/part-{part}.json"


def part_rel(title: str, part: str, date: str) -> str:
    return f"ecfr/full/{date}/title-{title}/part-{part}.xml"


def section_rel(title: str, part: str, section: str, date: str) -> str:
    return f"ecfr/full/{date}/title-{title}/part-{part}/section-{section}.xml"


def section_versions(versions_json: dict) -> list[dict]:
    """Section versions (no appendices), one row per (identifier, date).

    The API lists a version once per issue; reissues with substantive=false are editorial.
    A (identifier, date) pair is substantive if any of its issues is.
    """
    merged: dict[tuple[str, str], dict] = {}
    for v in versions_json["content_versions"]:
        if v["type"] != "section":
            continue
        key = (v["identifier"], v["date"])
        if key in merged:
            m = merged[key]
            m["substantive"] = m["substantive"] or v["substantive"]
            m["removed"] = m["removed"] or v["removed"]
            m["issue_date"] = max(m["issue_date"], v["issue_date"])
        else:
            merged[key] = {k: v[k] for k in
                           ("identifier", "date", "amendment_date", "issue_date", "name", "part",
                            "subpart", "substantive", "removed", "title")}
    return [merged[k] for k in sorted(merged, key=lambda k: (k[1], _section_sort_key(k[0])))]


def _section_sort_key(identifier: str) -> tuple:
    head, _, tail = identifier.partition(".")
    return (head, int(tail) if tail.isdigit() else 10**9, tail)


def history_fetch_list(scope: Scope, versions_json: dict) -> list[tuple[str, str]]:
    """(section, date) pairs whose text the version store needs beyond the part snapshots."""
    out = []
    for v in section_versions(versions_json):
        if v["substantive"] and not v["removed"] and scope.history_start < v["date"] <= scope.as_of:
            out.append((v["identifier"], v["date"]))
    return out


def ingest(scope: Scope, only: set[tuple[str, str]] | None = None, history: bool = True,
           dry_run: bool = False) -> dict:
    src = scope.raw["sources"]["ecfr"]
    base, lic = src["base_url"], src["license"]
    store = RawStore(scope.paths.raw, scope.raw.get("http"))
    counts = {"versions": 0, "parts": 0, "sections": 0}
    for title, part, _split in scope.parts():
        if only and (title, part) not in only:
            continue
        vrel = versions_rel(title, part)
        if dry_run and not store.has(vrel):
            log.info("would fetch %s", vrel)
            continue
        store.fetch(f"{base}/versions/title-{title}.json", vrel, license=lic, params={"part": part})
        counts["versions"] += 1
        for date in scope.snapshot_dates:
            rel = part_rel(title, part, date)
            if dry_run:
                counts["parts"] += not store.has(rel)
                continue
            if store.fetch(f"{base}/full/{date}/title-{title}.xml", rel, license=lic,
                           params={"part": part}, ok_missing=True):
                counts["parts"] += 1
        if not history:
            continue
        versions = json.loads(store.path(vrel).read_text())
        todo = history_fetch_list(scope, versions)
        log.info("%s CFR %s: %d section versions to fetch", title, part, len(todo))
        for section, date in todo:
            rel = section_rel(title, part, section, date)
            if dry_run:
                counts["sections"] += not store.has(rel)
                continue
            if store.fetch(f"{base}/full/{date}/title-{title}.xml", rel, license=lic,
                           params={"part": part, "section": section}, ok_missing=True):
                counts["sections"] += 1
    return counts


def _parse_only(values: list[str] | None) -> set[tuple[str, str]] | None:
    if not values:
        return None
    return {tuple(v.split(":", 1)) for v in values}  # type: ignore[misc]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--only", nargs="*", help="limit to title:part, e.g. 40:122")
    ap.add_argument("--no-history", action="store_true", help="snapshots only, skip per-version sections")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scope = load_scope(args.config, args.data_dir)
    counts = ingest(scope, _parse_only(args.only), history=not args.no_history, dry_run=args.dry_run)
    log.info("done: %s", counts)


if __name__ == "__main__":
    main()
