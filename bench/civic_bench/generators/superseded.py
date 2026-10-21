"""Superseded items from definition changes between two versions of a definitions section.

For each amended diff of a section with defined terms:
  - term defined in both versions with different text -> "redefined"
  - term defined in the old version only               -> "dropped"
Conditions: outdated_only (old version, dated) and outdated_plus_current (both, dated).
See bench/specs/superseded.md.
"""
from __future__ import annotations

import json
import random

from ..schema import Condition, Task, build_item, make_base_id
from .common import Tables, definition_text, passage, token_f1, window

GENERATOR = "superseded.definitions"
VERSION = "0.1"
MAX_SAME = 0.8    # token-F1 above this between old and new definitions counts as unchanged

TEMPLATES = {
    "sup-1": "As of {as_of}, how do the {topic} define \"{term}\"?",
    "sup-2": "Under the {topic} in force on {as_of}, what does \"{term}\" mean?",
    "sup-3": "What is the current ({as_of}) regulatory definition of \"{term}\" in the {topic}?",
    "sup-4": "On {as_of}, what counts as \"{term}\" under the {topic}?",
}


def _terms(ps: list[dict]) -> dict[str, int]:
    return {p["defined_term"]: i for i, p in enumerate(ps) if p.get("defined_term")}


def _in_force(versions, date: str):
    v = versions[versions.version_date <= date]
    return v.iloc[-1] if len(v) else None


def generate(t: Tables, *, n_base: int, seed: int, dataset_version: str) -> list:
    """Pairs (version in force at the reference date, version in force at as_of) per lineage.

    Reference date: test_cutoff for test parts (so the change is post-cutoff: time split),
    history_start for train/dev parts.
    """
    rng = random.Random(seed)
    found = []
    for lineage, vs in t.history.sort_values("version_date").groupby("lineage_id", sort=True):
        split = vs.split.iloc[0]
        ref = t.scope["test_cutoff"] if split == "test" else t.scope["history_start"]
        old, new = _in_force(vs, ref), _in_force(vs, t.as_of)
        if old is None or new is None or old.section_id == new.section_id or new.removed:
            continue
        ds = t.diffs[(t.diffs.lineage_id == lineage) & (t.diffs.effective_date > old.version_date)
                     & (t.diffs.effective_date <= new.version_date) & (t.diffs.change_type == "amended")]
        if ds.empty:
            continue
        d = ds.sort_values("effective_date").iloc[-1].to_dict()
        old = old.rename(old.section_id)
        new = new.rename(new.section_id)
        po, pn = json.loads(old.paragraphs), json.loads(new.paragraphs)
        to, tn = _terms(po), _terms(pn)
        for term, i in to.items():
            a_old = definition_text(po[i]["text"], term)
            if len(a_old.split()) < 5:
                continue
            if term in tn:
                a_new = definition_text(pn[tn[term]]["text"], term)
                if token_f1(a_old, a_new) >= MAX_SAME:
                    continue
                found.append(("redefined", term, d, old, new, po, pn, i, tn[term], a_old, a_new))
            else:
                found.append(("dropped", term, d, old, new, po, pn, i, None, a_old, None))
    rng.shuffle(found)
    # keep both kinds in the sample
    found.sort(key=lambda f: f[0] != "redefined")
    picked = found[:n_base] if len(found) <= n_base else (
        [f for f in found if f[0] == "redefined"][: n_base // 2]
        + [f for f in found if f[0] == "dropped"][: n_base - n_base // 2])

    items = []
    for k, (kind, term, d, old, new, po, pn, i_old, i_new, a_old, a_new) in enumerate(picked):
        old_row = {**old.to_dict(), "section_id": old.name}
        new_row = {**new.to_dict(), "section_id": new.name}
        tid = sorted(TEMPLATES)[k % len(TEMPLATES)]
        q = TEMPLATES[tid].format(as_of=t.as_of, topic=t.topic(old_row["title"], old_row["part"]), term=term)
        base_id = make_base_id(Task.superseded, d["lineage_id"], q, VERSION)
        label = {"outdated_section_id": old.name, "current_section_id": new.name, "change_type": "amended",
                 "outdated_version_date": old.version_date, "current_version_date": new.version_date,
                 "changed_paragraphs": [po[i_old]["label_path"]], "diff_id": d["diff_id"]}
        current_cit = f"{new_row['title']} CFR {new_row['section']}"
        common = {"base_id": base_id, "task": Task.superseded, "question": q, "as_of": t.as_of,
                  "split": d["split"], "lineage_id": d["lineage_id"], "title": old_row["title"],
                  "part": old_row["part"],
                  "provenance": {"generator": GENERATOR, "generator_version": VERSION, "seed": seed,
                                 "question_source": "template", "template_id": tid,
                                 "dataset_version": dataset_version,
                                 "source_hashes": [old.text_hash, new.text_hash]}}
        old_text, _ = window(po, i_old)
        p_old = passage("P1", old_row, old_text, "outdated", section_id=old.name, version_date=old.version_date)
        items.append(build_item(condition=Condition.outdated_only, passages=[p_old],
                                gold={"should_abstain": True, "superseded": label,
                                      "answer_aliases": [a_old]}, **common))
        new_text, _ = window(pn, i_new if i_new is not None else min(i_old, len(pn) - 1))
        p_new = passage("P2", new_row, new_text, "current", section_id=new.name, version_date=new.version_date)
        ps = [p_old, p_new]
        rng.shuffle(ps)
        ps = [p.model_copy(update={"passage_id": f"P{j + 1}"}) for j, p in enumerate(ps)]
        gold = ({"answer": a_new, "citations": [current_cit], "superseded": label, "answer_aliases": [a_old]}
                if kind == "redefined" else
                {"should_abstain": True, "superseded": label, "answer_aliases": [a_old]})
        items.append(build_item(condition=Condition.outdated_plus_current, passages=ps, gold=gold, **common))
    return items
