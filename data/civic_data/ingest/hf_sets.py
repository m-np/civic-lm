"""Stage 1c: pinned Hugging Face summarization sets (BillSum, GovReport) -> data/raw/hf/."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

from ..config import DEFAULT_CONFIG, load_scope
from ..http import PROVENANCE_LOG

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scope = load_scope(args.config, args.data_dir)
    for name, spec in scope.raw["sources"]["hf"].items():
        rel = f"hf/{name}@{spec['revision'][:12]}"
        dest = scope.paths.raw / rel
        if dest.exists():
            log.info("%s already present", rel)
            continue
        snapshot_download(repo_id=spec["repo"], repo_type="dataset", revision=spec["revision"],
                          local_dir=dest)
        with open(scope.paths.raw / PROVENANCE_LOG, "a") as f:
            f.write(json.dumps({
                "path": rel,
                "source_url": f"https://huggingface.co/datasets/{spec['repo']}/tree/{spec['revision']}",
                "license": spec["license"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sha256": spec["revision"],  # git revision pins the content
                "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()),
            }, sort_keys=True) + "\n")
        log.info("fetched %s", rel)


if __name__ == "__main__":
    main()
