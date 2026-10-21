"""Append-only fetcher for stage 1.

Every downloaded response is written once under data/raw/ and never overwritten.
Each write appends one row to data/raw/_provenance.jsonl.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .ids import sha256_bytes

PROVENANCE_LOG = "_provenance.jsonl"


class RawStore:
    def __init__(self, raw_dir: Path, http_cfg: dict | None = None, session: requests.Session | None = None):
        cfg = http_cfg or {}
        self.raw_dir = Path(raw_dir)
        self.min_interval = float(cfg.get("min_interval_s", 0.5))
        self.retries = int(cfg.get("retries", 4))
        self.timeout = float(cfg.get("timeout_s", 120))
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Accept-Encoding": "gzip, deflate", "User-Agent": "civic-lm-ingest/0.1 (research benchmark)"}
        )
        self._last = 0.0

    def path(self, rel: str) -> Path:
        return self.raw_dir / rel

    def has(self, rel: str) -> bool:
        return self.path(rel).exists()

    def fetch(self, url: str, rel: str, *, license: str, params: dict | None = None,
              ok_missing: bool = False) -> Path | None:
        """Download url to raw/rel unless it is already there. Returns the path, or None on 404
        when ok_missing (the 404 is recorded too, so reruns do not hit the API again)."""
        dest = self.path(rel)
        if dest.exists():
            return dest
        missing_marker = dest.with_name(dest.name + ".404")
        if missing_marker.exists():
            return None
        resp = self._get(url, params)
        if resp.status_code == 404 and ok_missing:
            missing_marker.parent.mkdir(parents=True, exist_ok=True)
            missing_marker.write_text(resp.url + "\n")
            return None
        resp.raise_for_status()
        self._write_once(dest, resp.content)
        self._log(url=resp.url, rel=rel, content=resp.content, license=license)
        return dest

    def _get(self, url: str, params: dict | None) -> requests.Response:
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.retries:
                    raise
            else:
                if resp.status_code not in (429, 500, 502, 503, 504):
                    return resp
                if attempt == self.retries:
                    return resp
            time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    @staticmethod
    def _write_once(dest: Path, content: bytes) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(content)
        os.replace(tmp, dest)

    def _log(self, *, url: str, rel: str, content: bytes, license: str) -> None:
        row = {
            "path": rel,
            "source_url": url,
            "license": license,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": sha256_bytes(content),
            "bytes": len(content),
        }
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with open(self.raw_dir / PROVENANCE_LOG, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
