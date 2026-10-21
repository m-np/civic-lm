"""Run one model x prompt tier over a benchmark release, logging every prompt and output.

    python -m civic_eval.run --release bench/releases/sample --model qwen2.5-1.5b --tier engineered

Output: eval/runs/<release>/<model>__<tier>/
  run.json        model id + revision, backend, generation settings, release hash, prompt version
  outputs.jsonl   one row per item: messages, raw output, tokens, latency
Re-running resumes: items already in outputs.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import yaml
from civic_bench.schema import Item

from .backends import BACKENDS
from .prompts import PROMPT_VERSION, TIERS, build_messages

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_release(path: Path) -> tuple[list[Item], dict, dict]:
    items = [Item.model_validate_json(line) for line in open(path / "items.jsonl")]
    return items, json.loads((path / "passage_validity.json").read_text()), \
        json.loads((path / "MANIFEST.json").read_text())


def run_dir(release: Path, model_key: str, tier: str) -> Path:
    return REPO_ROOT / "eval" / "runs" / release.name / f"{model_key}__{tier}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--model", required=True, help="key in configs/models.yaml")
    ap.add_argument("--tier", choices=TIERS, default="engineered")
    ap.add_argument("--models-config", type=Path, default=REPO_ROOT / "configs" / "models.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="only items of this split")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.models_config))
    spec, gen = cfg["models"][args.model], cfg["generation"]
    items, validity, manifest = load_release(args.release)
    if args.split:
        items = [i for i in items if i.split == args.split]
    items = items[: args.limit] if args.limit else items

    out = run_dir(args.release, args.model, args.tier)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model_key": args.model, **spec, "generation": gen, "tier": args.tier,
            "prompt_version": PROMPT_VERSION, "release": args.release.name,
            "release_items_sha256": manifest["items_sha256"], "host": platform.node(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    done = set()
    if (out / "outputs.jsonl").exists():
        done = {json.loads(line)["item_id"] for line in open(out / "outputs.jsonl")}
    todo = [i for i in items if i.item_id not in done]
    print(f"{args.model}/{args.tier}: {len(todo)} to run ({len(done)} already done)", flush=True)
    if not todo:
        return
    backend = BACKENDS[spec["backend"]](model=spec["model"], revision=spec.get("revision"),
                                        seed=gen["seed"], max_new_tokens=gen["max_new_tokens"])
    with open(out / "outputs.jsonl", "a") as f:
        for n, item in enumerate(todo, 1):
            msgs = build_messages(item, args.tier, validity)
            g = backend.generate(msgs)
            f.write(json.dumps({"item_id": item.item_id, "messages": msgs, "output": g.text,
                                "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                                "latency_s": round(g.latency_s, 3)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{n}/{len(todo)}] {item.task.value}/{item.condition.value} {g.latency_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
