"""Benchmark item schema: the one contract shared by bench, eval and train.

eval and train import only this module from civic_bench; everything else they get from
the released JSONL/Parquet files.
"""
from __future__ import annotations

import hashlib
import json
import re
from enum import Enum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

SECTION_ID_RE = re.compile(r"^\d+-CFR-[0-9A-Za-z.\-]+@\d{4}-\d{2}-\d{2}$")
LINEAGE_ID_RE = re.compile(r"^\d+-CFR-[0-9A-Za-z.\-]+$")
CITATION_RE = re.compile(r"^\d+ CFR \d+[A-Za-z]?\.\d+[A-Za-z0-9\-]*(\([A-Za-z0-9]+\))*$")
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"


class Task(StrEnum):
    grounded_qa = "grounded_qa"
    abstention = "abstention"
    superseded = "superseded"
    summary = "summary"
    obligation = "obligation"
    closed_book = "closed_book"


class Condition(StrEnum):
    gold_only = "gold_only"                    # the gold passage alone
    gold_distractors = "gold_distractors"      # gold passage + hard distractors, shuffled
    no_gold = "no_gold"                        # only near-miss passages; correct answer is to abstain
    outdated_only = "outdated_only"            # only a superseded version; must flag it
    outdated_plus_current = "outdated_plus_current"  # both versions, dated; must use the current one
    closed_book = "closed_book"                # no passages


# Which conditions each task may appear under.
ALLOWED_CONDITIONS: dict[Task, set[Condition]] = {
    Task.grounded_qa: {Condition.gold_only, Condition.gold_distractors},
    Task.abstention: {Condition.no_gold},
    Task.superseded: {Condition.outdated_only, Condition.outdated_plus_current},
    Task.summary: {Condition.gold_only},
    Task.obligation: {Condition.gold_only},
    Task.closed_book: {Condition.closed_book},
}

Split = Literal["train", "dev", "test"]
PassageRole = Literal["gold", "distractor", "near_miss", "outdated", "current"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Passage(Strict):
    passage_id: str = Field(description="Label shown to the model, e.g. 'P1'.")
    section_id: str | None = Field(default=None, description="Set for CFR passages.")
    doc_id: str | None = Field(default=None, description="Set for non-CFR documents (bills, reports).")
    citation: str | None = None
    version_date: str | None = Field(default=None, pattern=DATE_RE)
    heading: str = ""
    text: str
    role: PassageRole

    @model_validator(mode="after")
    def _ids(self):
        if (self.section_id is None) == (self.doc_id is None):
            raise ValueError("exactly one of section_id / doc_id must be set")
        if self.section_id and not SECTION_ID_RE.match(self.section_id):
            raise ValueError(f"bad section_id {self.section_id!r}")
        if self.citation and not CITATION_RE.match(self.citation):
            raise ValueError(f"bad citation {self.citation!r}")
        return self


class SupersededLabel(Strict):
    outdated_section_id: str
    current_section_id: str | None      # None when the section was removed
    change_type: Literal["amended", "removed", "renumbered"]
    outdated_version_date: str = Field(pattern=DATE_RE)
    current_version_date: str | None = Field(default=None, pattern=DATE_RE)
    changed_paragraphs: list[str] = Field(default_factory=list)
    diff_id: str


class Obligation(Strict):
    actor: str
    action: str
    modality: Literal["must", "must_not", "may"]
    condition: str | None = None
    deadline: str | None = None
    citation: str


class Gold(Strict):
    answer: str | None = None
    answer_aliases: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list, description="Gold citations, normalized.")
    acceptable_citations: list[str] = Field(
        default_factory=list, description="Also correct but not required (e.g. the parent section).")
    should_abstain: bool = False
    superseded: SupersededLabel | None = None
    reference_summary: str | None = None
    obligations: list[Obligation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _citations(self):
        for c in self.citations + self.acceptable_citations:
            if not CITATION_RE.match(c):
                raise ValueError(f"bad citation {c!r}")
        return self


class Provenance(Strict):
    generator: str
    generator_version: str
    seed: int
    question_source: Literal["template", "llm", "dataset", "human"]
    question_model: str | None = Field(default=None, description="Model that wrote the question, if any.")
    template_id: str | None = None
    source_hashes: list[str] = Field(default_factory=list)
    dataset_version: str


class Item(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    item_id: str
    base_id: str = Field(description="Shared by all conditions built from the same base question.")
    task: Task
    condition: Condition
    question: str
    as_of: str = Field(pattern=DATE_RE, description="The date the question asks about ('current rule').")
    passages: list[Passage] = Field(default_factory=list)
    gold: Gold
    split: Split
    lineage_id: str = Field(description="Lineage of the gold section; the dedupe and split key.")
    title: str
    part: str
    provenance: Provenance
    human_verified: bool = False
    verification_notes: str | None = None

    @model_validator(mode="after")
    def _consistency(self):
        if self.condition not in ALLOWED_CONDITIONS[self.task]:
            raise ValueError(f"{self.task.value} cannot use condition {self.condition.value}")
        if not LINEAGE_ID_RE.match(self.lineage_id) and self.task != Task.summary:
            raise ValueError(f"bad lineage_id {self.lineage_id!r}")
        ids = [p.passage_id for p in self.passages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate passage_id")
        roles = [p.role for p in self.passages]
        c, g = self.condition, self.gold
        if c == Condition.closed_book and self.passages:
            raise ValueError("closed_book items carry no passages")
        if c == Condition.gold_only and roles != ["gold"]:
            raise ValueError("gold_only needs exactly one gold passage")
        if c == Condition.gold_distractors and (roles.count("gold") != 1 or len(roles) < 2):
            raise ValueError("gold_distractors needs one gold passage and >=1 distractor")
        if c == Condition.no_gold and ("gold" in roles or not roles or not g.should_abstain):
            raise ValueError("no_gold needs passages, none gold, and should_abstain=True")
        if c in (Condition.outdated_only, Condition.outdated_plus_current):
            if g.superseded is None or "outdated" not in roles:
                raise ValueError("superseded items need an outdated passage and a superseded label")
            if c == Condition.outdated_plus_current and "current" not in roles:
                raise ValueError("outdated_plus_current needs the current passage too")
            if c == Condition.outdated_only and "current" in roles:
                raise ValueError("outdated_only must not include the current passage")
        if self.task in (Task.grounded_qa, Task.closed_book) and not (g.answer and g.citations):
            raise ValueError(f"{self.task.value} needs a gold answer and citations")
        if self.task == Task.summary and not g.reference_summary:
            raise ValueError("summary needs reference_summary")
        if self.task == Task.obligation and not g.obligations:
            raise ValueError("obligation needs gold obligations")
        if self.task != Task.abstention and c != Condition.no_gold and g.should_abstain \
                and self.task != Task.superseded:
            raise ValueError("only abstention / superseded items may expect abstention")
        if self.item_id != make_item_id(self):
            raise ValueError("item_id does not match content; build items with make_item_id")
        return self


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_base_id(task: Task | str, lineage_id: str, question: str, generator_version: str) -> str:
    t = task.value if isinstance(task, Task) else task
    h = hashlib.sha256(_canonical([t, lineage_id, question, generator_version]).encode()).hexdigest()
    return f"{t}-{h[:12]}"


def _plain(v):
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json")
    if isinstance(v, Enum):
        return v.value
    return v


def make_item_id(item: Item | dict) -> str:
    """Deterministic id from content: the same inputs always give the same id."""
    d = item.model_dump(mode="json") if isinstance(item, Item) else {k: _plain(v) for k, v in item.items()}
    passages = [_plain(p) for p in d.get("passages", [])]
    key = [d["base_id"], d["condition"], [p.get("section_id") or p.get("doc_id") for p in passages],
           _plain(d["provenance"])["seed"]]
    h = hashlib.sha256(_canonical(key).encode()).hexdigest()
    return f"{d['base_id']}-{d['condition']}-{h[:8]}"


def build_item(**fields) -> Item:
    """Construct an Item, filling item_id from content."""
    return Item(**fields, item_id=make_item_id(fields))


# ---------------------------------------------------------------- model output contract

class ModelAnswer(Strict):
    """What every prompt asks the model to return (JSON). Scoring parses into this."""
    model_config = ConfigDict(extra="ignore", frozen=True)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    abstain: bool = False
    outdated_warning: bool = False
