"""Stage 1b: Federal Register API -> data/raw/fedreg/.

Final rules that touch each in-scope part, published on or after history_start.
Used to attach amending documents (and effective dates) to diffs, and to confirm renumberings.
"""
from __future__ import annotations

import argparse
import json
import logging

from ..config import DEFAULT_CONFIG, Scope, load_scope
from ..http import RawStore

log = logging.getLogger(__name__)

FIELDS = ["document_number", "title", "type", "publication_date", "effective_on",
          "cfr_references", "html_url", "raw_text_url", "agencies", "action", "citation"]


def page_rel(title: str, part: str, page: int) -> str:
    return f"fedreg/title-{title}/part-{part}/page-{page:03d}.json"


def ingest(scope: Scope, only: set[tuple[str, str]] | None = None) -> int:
    src = scope.raw["sources"]["fedreg"]
    store = RawStore(scope.paths.raw, scope.raw.get("http"))
    n = 0
    for title, part, _ in scope.parts():
        if only and (title, part) not in only:
            continue
        params = {
            "per_page": 1000,
            "order": "oldest",
            "conditions[cfr][title]": title,
            "conditions[cfr][part]": part,
            "conditions[type][]": "RULE",
            "conditions[publication_date][gte]": scope.history_start,
            "conditions[publication_date][lte]": scope.as_of,
            "fields[]": FIELDS,
        }
        page = 1
        while True:
            params["page"] = page
            path = store.fetch(f"{src['base_url']}/documents.json", page_rel(title, part, page),
                               license=src["license"], params=params)
            n += 1
            body = json.loads(path.read_text())
            if page >= int(body.get("total_pages") or 1):
                break
            page += 1
        log.info("%s CFR %s: %d page(s)", title, part, page)
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    only = {tuple(v.split(":", 1)) for v in args.only} if args.only else None
    ingest(load_scope(args.config, args.data_dir), only)


if __name__ == "__main__":
    main()
