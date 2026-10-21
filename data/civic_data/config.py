"""Load configs/scope.yaml and resolve repo-relative paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "scope.yaml"
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path

    @classmethod
    def under(cls, data_dir: Path) -> Paths:
        return cls(data_dir / "raw", data_dir / "interim", data_dir / "processed")


@dataclass(frozen=True)
class Scope:
    raw: dict
    paths: Paths

    @property
    def as_of(self) -> str:
        return self.raw["as_of"]

    @property
    def history_start(self) -> str:
        return self.raw["history_start"]

    @property
    def test_cutoff(self) -> str:
        return self.raw["test_cutoff"]

    @property
    def snapshot_dates(self) -> list[str]:
        """Dates at which whole parts are snapshotted: baseline, time-split boundary, now."""
        return sorted({self.history_start, self.test_cutoff, self.as_of})

    def parts(self) -> list[tuple[str, str, str]]:
        """(title, part, split) for every part in scope, in a stable order."""
        out = []
        for title, by_split in sorted(self.raw["titles"].items(), key=lambda kv: int(kv[0])):
            for split in SPLITS:
                for part in by_split.get(split, []):
                    out.append((str(title), str(part), split))
        return out

    def split_of(self, title: str, part: str) -> str | None:
        for t, p, s in self.parts():
            if t == str(title) and p == str(part):
                return s
        return None


def load_scope(path: str | Path = DEFAULT_CONFIG, data_dir: str | Path | None = None) -> Scope:
    with open(path) as f:
        raw = yaml.safe_load(f)
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / "data"
    scope = Scope(raw=raw, paths=Paths.under(data_dir))
    _check_parts_unique(scope)
    return scope


def _check_parts_unique(scope: Scope) -> None:
    seen: dict[tuple[str, str], str] = {}
    for t, p, s in scope.parts():
        if (t, p) in seen:
            raise ValueError(f"{t} CFR part {p} is in both {seen[(t, p)]} and {s}")
        seen[(t, p)] = s
