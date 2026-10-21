"""Classify changes between consecutive versions of a section.

Classes (from the plan): added | amended | removed | renumbered.
Plus "editorial": the text changed but the versioner API marks the version non-substantive.
Editorial rows are kept for audit; generators must not build items from them.
"""
from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import asdict, dataclass, field

CHANGE_TYPES = ("added", "amended", "removed", "renumbered", "editorial")
RENUMBER_MIN_SIMILARITY = 0.85
_MAX_WORDS_FOR_SEQMATCH = 20_000


@dataclass
class Version:
    lineage_id: str
    section_id: str
    title: str
    part: str
    section: str
    version_date: str
    text: str
    text_hash: str
    paragraphs: list[dict]
    substantive: bool = True
    removed: bool = False
    reserved: bool = False

    @property
    def gone(self) -> bool:
        return self.removed or self.reserved


@dataclass
class Diff:
    diff_id: str
    change_type: str
    title: str
    part: str
    effective_date: str
    lineage_id: str                  # lineage after the change (target for renumbering)
    from_section_id: str | None
    to_section_id: str | None
    from_lineage_id: str | None = None
    similarity: float | None = None
    changed_paragraphs: list[dict] = field(default_factory=list)
    fr_documents: list[str] = field(default_factory=list)


def similarity(a: str, b: str) -> float:
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 1.0
    if len(wa) + len(wb) > _MAX_WORDS_FOR_SEQMATCH:
        sa, sb = set(wa), set(wb)
        return len(sa & sb) / max(1, len(sa | sb))
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def paragraph_changes(old: list[dict], new: list[dict]) -> list[dict]:
    """Per-label-path changes. Unlabeled continuation text is folded into its path."""
    def by_path(ps):
        d: dict[str, list[str]] = defaultdict(list)
        for p in ps:
            d[p["label_path"]].append(" ".join(p["text"].split()))
        return {k: "\n".join(v) for k, v in d.items()}

    a, b = by_path(old), by_path(new)
    out = []
    for path in sorted(set(a) | set(b), key=lambda p: (p not in b, p)):
        if path not in a:
            out.append({"label_path": path, "change": "added"})
        elif path not in b:
            out.append({"label_path": path, "change": "removed"})
        elif a[path] != b[path]:
            out.append({"label_path": path, "change": "modified"})
    return out


def _diff_id(change_type: str, lineage: str, date: str) -> str:
    return f"{lineage}@{date}:{change_type}"


def classify_lineage(versions: list[Version], *, history_start: str,
                     part_first_date: str | None) -> list[Diff]:
    """Diffs for one lineage. `versions` must be sorted by version_date.

    part_first_date is the earliest version date anywhere in the part. A lineage whose
    first version is later than both that and history_start was added in the window;
    one present from the start of eCFR history is the baseline, not an addition.
    """
    out: list[Diff] = []
    if not versions:
        return out
    first = versions[0]
    if (first.version_date > history_start and part_first_date is not None
            and first.version_date > part_first_date and not first.gone):
        out.append(Diff(_diff_id("added", first.lineage_id, first.version_date), "added",
                        first.title, first.part, first.version_date, first.lineage_id,
                        None, first.section_id,
                        changed_paragraphs=paragraph_changes([], first.paragraphs)))
    for prev, cur in zip(versions, versions[1:], strict=False):
        if cur.version_date <= history_start:
            continue
        if cur.gone and not prev.gone:
            ctype = "removed"
        elif prev.gone and not cur.gone:
            ctype = "added"
        elif cur.gone and prev.gone:
            continue
        elif cur.text_hash == prev.text_hash:
            continue
        elif not cur.substantive:
            ctype = "editorial"
        else:
            ctype = "amended"
        out.append(Diff(
            _diff_id(ctype, cur.lineage_id, cur.version_date), ctype, cur.title, cur.part,
            cur.version_date, cur.lineage_id, prev.section_id,
            None if ctype == "removed" and cur.removed else cur.section_id,
            similarity=None if ctype in ("added", "removed") else round(similarity(prev.text, cur.text), 4),
            changed_paragraphs=paragraph_changes(prev.paragraphs, [] if cur.gone else cur.paragraphs),
        ))
    return out


def pair_renumberings(diffs: list[Diff], text_of: dict[str, str]) -> list[Diff]:
    """Replace a same-day (removed X, added Y) pair in one part with one renumbered X->Y
    diff when the texts are nearly identical. Greedy on best similarity.

    text_of maps section_id -> text; a removed diff's old text is its from_section_id.
    """
    by_key: dict[tuple, dict[str, list[Diff]]] = defaultdict(lambda: {"removed": [], "added": []})
    for d in diffs:
        if d.change_type in ("removed", "added"):
            by_key[(d.title, d.part, d.effective_date)][d.change_type].append(d)
    consumed: set[str] = set()
    new: list[Diff] = []
    for (title, part, date), g in sorted(by_key.items()):
        cands = []
        for r in g["removed"]:
            for a in g["added"]:
                s = similarity(text_of.get(r.from_section_id or "", ""), text_of.get(a.to_section_id or "", ""))
                if s >= RENUMBER_MIN_SIMILARITY:
                    cands.append((s, r, a))
        for s, r, a in sorted(cands, key=lambda c: (-c[0], c[1].diff_id, c[2].diff_id)):
            if r.diff_id in consumed or a.diff_id in consumed:
                continue
            consumed |= {r.diff_id, a.diff_id}
            new.append(Diff(_diff_id("renumbered", a.lineage_id, date), "renumbered", title, part,
                            date, a.lineage_id, r.from_section_id, a.to_section_id,
                            from_lineage_id=r.lineage_id, similarity=round(s, 4)))
    kept = [d for d in diffs if d.diff_id not in consumed]
    return sorted(kept + new, key=lambda d: (d.title, d.part, d.effective_date, d.lineage_id, d.change_type))


def diffs_to_rows(diffs: list[Diff]) -> list[dict]:
    return [asdict(d) for d in diffs]
