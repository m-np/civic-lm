"""Content hashes of processed tables, for the "rebuild from raw is reproducible" gate.

Hashes cover canonical row content (sorted JSON), not Parquet bytes, so they do not depend
on the pyarrow version that wrote the file.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .config import DEFAULT_CONFIG, Scope, load_scope
from .ids import sha256_text

MANIFEST = "MANIFEST.json"
# Columns that legitimately differ between two ingests of the same content.
VOLATILE = {"provenance": ["retrieved_at"]}


def content_hash(name: str, df: pd.DataFrame) -> str:
    df = df.drop(columns=[c for c in VOLATILE.get(name, []) if c in df.columns])
    return sha256_text(df.to_json(orient="records", force_ascii=False, date_format="iso"))


def write_tables(scope: Scope, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    out = scope.paths.processed
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        manifest[name] = {"rows": len(df), "sha256": content_hash(name, df),
                          "dataset_version": scope.raw["dataset_version"]}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {k: v["sha256"] for k, v in manifest.items()}


def check(scope: Scope, expected_path: str | None) -> list[str]:
    """Compare the current tables with a recorded manifest; returns the names that differ."""
    out = scope.paths.processed
    expected = json.loads(open(expected_path or out / MANIFEST).read())
    bad = []
    for name, rec in expected.items():
        p = out / f"{name}.parquet"
        if not p.exists() or content_hash(name, pd.read_parquet(p)) != rec["sha256"]:
            bad.append(name)
    return bad


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--expected", default="configs/data_manifest.lock.json",
                    help="recorded manifest to compare against")
    args = ap.parse_args(argv)
    scope = load_scope(args.config, args.data_dir)
    if args.check:
        bad = check(scope, args.expected)
        if bad:
            print(f"MISMATCH: {', '.join(bad)}", file=sys.stderr)
            sys.exit(1)
        print("manifest OK")


if __name__ == "__main__":
    main()
