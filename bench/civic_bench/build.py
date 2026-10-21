"""Build a benchmark release from data/processed/ -> bench/releases/<name>/.

    python -m civic_bench.build --name sample --n-definitions 12 --n-superseded 8

Writes items.jsonl (one Item per line), valid_sections.json (every section that exists in the
processed data, for the closed-book hallucinated-citation metric), passage_validity.json
(effective-from/to per passage version, used to render dated passages), and MANIFEST.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .generators import definitions, superseded
from .generators.common import REPO_ROOT, Tables

DEFAULT_CONFIG = REPO_ROOT / "configs" / "scope.yaml"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="sample")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--processed", default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--out", default=REPO_ROOT / "bench" / "releases")
    ap.add_argument("--n-definitions", type=int, default=12)
    ap.add_argument("--n-superseded", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)

    t = Tables.load(Path(args.config), Path(args.processed))
    version = t.scope["dataset_version"]
    items = definitions.generate(t, n_base=args.n_definitions, seed=args.seed, dataset_version=version)
    items += superseded.generate(t, n_base=args.n_superseded, seed=args.seed, dataset_version=version)
    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w") as f:
        for it in sorted(items, key=lambda i: i.item_id):
            f.write(it.model_dump_json() + "\n")

    valid = sorted({f"{r.title} CFR {r.section}" for r in t.sections.itertuples()}
                   | {f"{r.title} CFR {r.section}" for r in t.history.itertuples()})
    (out / "valid_sections.json").write_text(json.dumps(valid, indent=0))
    validity = {}
    for it in items:
        for p in it.passages:
            if p.section_id:
                validity[p.section_id] = {"from": p.version_date, "to": t.valid_to(p.section_id)}
    (out / "passage_validity.json").write_text(json.dumps(validity, indent=1, sort_keys=True))

    digest = hashlib.sha256((out / "items.jsonl").read_bytes()).hexdigest()
    counts = Counter(f"{i.task.value}/{i.condition.value}" for i in items)
    manifest = {"name": args.name, "dataset_version": version, "seed": args.seed, "items": len(items),
                "items_sha256": digest, "by_task_condition": dict(sorted(counts.items())),
                "by_split": dict(Counter(i.split for i in items)), "as_of": t.as_of}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
