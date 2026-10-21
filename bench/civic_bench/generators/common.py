"""Shared helpers for generators: loading processed tables, passage windows, lexical ranking."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..schema import Passage

REPO_ROOT = Path(__file__).resolve().parents[3]
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of to in on for by or and is are be any as at with from that this which
such under shall may must means mean its it their than other not no all each""".split())
_LEAD_LABELS = re.compile(r"^\s*(?:\([A-Za-z0-9]{1,6}\)\s*)+")

WINDOW_WORDS = 220      # target size of a passage built around one paragraph


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def token_f1(pred: str, gold: str) -> float:
    p, g = Counter(tokens(pred)), Counter(tokens(gold))
    common = sum((p & g).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


@dataclass
class Tables:
    scope: dict
    sections: pd.DataFrame
    history: pd.DataFrame
    diffs: pd.DataFrame

    @classmethod
    def load(cls, config: Path, processed: Path) -> Tables:
        return cls(scope=yaml.safe_load(open(config)),
                   sections=pd.read_parquet(processed / "sections.parquet"),
                   history=pd.read_parquet(processed / "section_history.parquet"),
                   diffs=pd.read_parquet(processed / "diffs.parquet"))

    @property
    def as_of(self) -> str:
        return self.scope["as_of"]

    def current(self) -> pd.DataFrame:
        s = self.sections
        return s[(s.snapshot_date == self.as_of) & (~s.reserved) & (s.text.str.len() > 0)]

    def topic(self, title: str, part: str) -> str:
        return self.scope.get("part_topics", {}).get(f"{title}:{part}", f"Title {title} regulations")

    def valid_to(self, section_id: str) -> str | None:
        """Date a version stopped being in force (the next version's date), None if current."""
        lineage, date = section_id.split("@")
        later = self.history[(self.history.lineage_id == lineage) & (self.history.version_date > date)]
        return later.version_date.min() if len(later) else None


def paragraphs(row) -> list[dict]:
    return json.loads(row["paragraphs"])


def strip_labels(text: str) -> str:
    return _LEAD_LABELS.sub("", text).strip()


def definition_text(paragraph_text: str, term: str) -> str:
    """'(a) Act means the Federal ...' -> 'the Federal ...'."""
    t = strip_labels(paragraph_text).lstrip("“\"")
    if t.startswith(term):
        t = t[len(term):]
    t = t.lstrip("”\" ")
    t = re.sub(r"^\([^)]*\)\s*", "", t)  # "(DHF)"
    t = re.sub(r"^(means|is defined as|includes|has the meaning given|refers to)\s+", "", t)
    return t.strip()


def window(paras: list[dict], center: int, *, exclude: set[int] = frozenset(),
           words: int = WINDOW_WORDS) -> tuple[str, list[int]]:
    """Text of paragraphs around `center` (inclusive) up to about `words` words."""
    picked = [center] if center not in exclude else []
    budget = words - len(paras[center]["text"].split()) if picked else words
    lo, hi = center - 1, center + 1
    while budget > 0 and (lo >= 0 or hi < len(paras)):
        for j in (hi, lo):
            if 0 <= j < len(paras) and j not in exclude and j not in picked:
                n = len(paras[j]["text"].split())
                if n <= budget or not picked:
                    picked.append(j)
                    budget -= n
        lo, hi = lo - 1, hi + 1
    picked.sort()
    return "\n".join(paras[j]["text"] for j in picked), picked


def citation_of(row, label_path: str = "") -> str:
    return f"{row['title']} CFR {row['section']}{label_path}"


def passage(pid: str, row, text: str, role: str, *, section_id: str | None = None,
            version_date: str | None = None) -> Passage:
    return Passage(passage_id=pid, section_id=section_id or row["section_id"],
                   citation=citation_of(row), version_date=version_date or row["version_date"],
                   heading=row["heading"], text=text, role=role)


class LexicalIndex:
    """Small BM25 over paragraph windows; enough for picking hard distractors in a pilot.
    (The full build swaps in bm25s; same interface.)"""

    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.2, b: float = 0.75):
        self.ids = [d[0] for d in docs]
        self.toks = [tokens(d[1]) for d in docs]
        self.df = Counter(t for ts in self.toks for t in set(ts))
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        self.k1, self.b, self.n = k1, b, len(docs)

    def search(self, query: str, k: int, skip=lambda doc_id: False) -> list[str]:
        q = tokens(query)
        scored = []
        for i, ts in enumerate(self.toks):
            if skip(self.ids[i]):
                continue
            tf = Counter(ts)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(ts) / self.avg))
            if s > 0:
                scored.append((s, self.ids[i]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [d for _, d in scored[:k]]
