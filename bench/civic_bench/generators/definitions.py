"""Definition-based items: grounded_qa (gold_only, gold_distractors), abstention (no_gold),
closed_book. No LLM: the question comes from a template, the gold answer and citation from
the definition paragraph itself. See bench/specs/grounded_qa.md (source A)."""
from __future__ import annotations

import random

from ..schema import Condition, Task, build_item, make_base_id
from .common import LexicalIndex, Tables, citation_of, definition_text, paragraphs, passage, token_f1, window

GENERATOR = "definitions"
VERSION = "0.1"

TEMPLATES = {
    "def-1": "Under the {topic}, what does the term \"{term}\" mean?",
    "def-2": "How do the {topic} define \"{term}\"?",
    "def-3": "According to the {topic}, what is meant by \"{term}\"?",
    "def-4": "What is the regulatory definition of \"{term}\" in the {topic}?",
    "def-5": "In the {topic}, what counts as \"{term}\"?",
}


def candidates(t: Tables) -> list[dict]:
    """Every usable defined term in current sections: one dict per (section, term)."""
    out = []
    for row in t.current().to_dict("records"):
        ps = paragraphs(row)
        for i, p in enumerate(ps):
            term = p.get("defined_term")
            if not term:
                continue
            answer = definition_text(p["text"], term)
            # cross-references ("is defined at § 122.23") carry no answer
            if len(answer.split()) < 5 or answer.lower().startswith(("defined at", "at §", "is defined")):
                continue
            out.append({"row": row, "idx": i, "term": term, "answer": answer, "paras": ps})
    return out


def _index(t: Tables) -> tuple[LexicalIndex, dict]:
    docs, meta = [], {}
    for row in t.current().to_dict("records"):
        ps = paragraphs(row)
        for i in range(0, len(ps), 3):
            text, _ = window(ps, i, words=150)
            doc_id = f"{row['section_id']}#{i}"
            docs.append((doc_id, text))
            meta[doc_id] = (row, text, ps[i:i + 3])
    return LexicalIndex(docs), meta


def _mentions_definition(text: str, term: str, answer: str) -> bool:
    return term.lower() in text.lower() and token_f1(text, answer) > 0.5


def generate(t: Tables, *, n_base: int, seed: int, dataset_version: str) -> list:
    rng = random.Random(seed)
    cands = candidates(t)
    # stratify by part so one long definitions section does not dominate
    by_part: dict[str, list] = {}
    for c in cands:
        by_part.setdefault(f"{c['row']['title']}:{c['row']['part']}", []).append(c)
    picked = []
    for key in sorted(by_part):
        rng.shuffle(by_part[key])
    while len(picked) < n_base and any(by_part.values()):
        for key in sorted(by_part):
            if by_part[key] and len(picked) < n_base:
                picked.append(by_part[key].pop())
    index, meta = _index(t)
    template_ids = sorted(TEMPLATES)
    items = []
    for k, c in enumerate(picked):
        row, term = c["row"], c["term"]
        tid = template_ids[k % len(template_ids)]
        q = TEMPLATES[tid].format(topic=t.topic(row["title"], row["part"]), term=term)
        label = c["paras"][c["idx"]]["label_path"] if c["paras"][c["idx"]]["labeled"] else ""
        gold_cits = [citation_of(row, label)] if label else [citation_of(row)]
        acceptable = [citation_of(row)] if label else []
        base_id = make_base_id(Task.grounded_qa, row["lineage_id"], q, VERSION)
        prov = {"generator": GENERATOR, "generator_version": VERSION, "seed": seed,
                "question_source": "template", "template_id": tid, "dataset_version": dataset_version,
                "source_hashes": [row["text_hash"]]}
        common = {"base_id": base_id, "question": q, "as_of": t.as_of, "split": row["split"],
                  "lineage_id": row["lineage_id"], "title": row["title"], "part": row["part"],
                  "provenance": prov}
        gold = {"answer": c["answer"], "citations": gold_cits, "acceptable_citations": acceptable}
        gold_text, _ = window(c["paras"], c["idx"])
        gold_p = passage("P1", row, gold_text, "gold")

        items.append(build_item(task=Task.grounded_qa, condition=Condition.gold_only,
                                passages=[gold_p], gold=gold, **common))

        # hard distractors: lexically close to the term and its definition, other sections only,
        # at most one window per section, and none that defines the term
        own = row["lineage_id"] + "@"
        hits = index.search(f"{term} {term} {c['answer']}", k=30, skip=lambda d, own=own: d.startswith(own))
        distract, used = [], set()
        for h in hits:
            sec = h.split("#")[0]
            if sec in used or _mentions_definition(meta[h][1], term, c["answer"]):
                continue
            used.add(sec)
            distract.append(h)
            if len(distract) == 3:
                break
        if distract:
            ps = [gold_p] + [passage("P0", meta[h][0], meta[h][1], "distractor") for h in distract]
            rng.shuffle(ps)
            ps = [p.model_copy(update={"passage_id": f"P{i + 1}"}) for i, p in enumerate(ps)]
            items.append(build_item(task=Task.grounded_qa, condition=Condition.gold_distractors,
                                    passages=ps, gold=gold, **common))

        # abstention: the same section with the defining paragraph removed, plus a neighbor
        near_text, _ = window(c["paras"], c["idx"], exclude={c["idx"]})
        near = [passage("P1", row, near_text, "near_miss")] if near_text else []
        near += [passage("P2", meta[h][0], meta[h][1], "near_miss") for h in distract[:1]]
        if near and not any(_mentions_definition(p.text, term, c["answer"]) for p in near):
            items.append(build_item(task=Task.abstention, condition=Condition.no_gold, passages=near,
                                    gold={"should_abstain": True},
                                    **{**common, "base_id": base_id.replace("grounded_qa", "abstention", 1)}))

        if k % 2 == 0:  # closed-book control on half of the base questions
            cb_id = base_id.replace("grounded_qa", "closed_book", 1)
            items.append(build_item(task=Task.closed_book, condition=Condition.closed_book, passages=[],
                                    gold=gold, **{**common, "base_id": cb_id}))
    return items
