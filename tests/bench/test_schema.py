import pytest
from civic_bench.schema import Condition, Item, Task, build_item, make_base_id
from pydantic import ValidationError


def _prov(**kw):
    return {"generator": "qa.definitions", "generator_version": "0.1", "seed": 7,
            "question_source": "template", "template_id": "def-1", "dataset_version": "0.1.0-pilot", **kw}


def _passage(role="gold", sec="122.2", pid="P1"):
    return {"passage_id": pid, "section_id": f"40-CFR-{sec}@2019-02-12", "citation": f"40 CFR {sec}",
            "version_date": "2019-02-12", "text": "Administrator means ...", "role": role}


def _qa(condition=Condition.gold_only, passages=None, **kw):
    q = "Under the NPDES rules, who is the Administrator?"
    base = dict(base_id=make_base_id(Task.grounded_qa, "40-CFR-122.2", q, "0.1"), task=Task.grounded_qa,
                condition=condition, question=q, as_of="2026-10-19",
                passages=passages if passages is not None else [_passage()],
                gold={"answer": "The EPA Administrator or an authorized representative.",
                      "citations": ["40 CFR 122.2"]},
                split="train", lineage_id="40-CFR-122.2", title="40", part="122", provenance=_prov())
    base.update(kw)
    return base


def test_valid_item_roundtrip():
    it = build_item(**_qa())
    again = Item.model_validate_json(it.model_dump_json())
    assert again == it
    assert it.item_id.startswith(it.base_id + "-gold_only-")


def test_item_id_is_deterministic_and_content_sensitive():
    a, b = build_item(**_qa()), build_item(**_qa())
    assert a.item_id == b.item_id
    c = build_item(**_qa(provenance=_prov(seed=8)))
    assert c.item_id != a.item_id


def test_tampered_item_id_rejected():
    d = build_item(**_qa()).model_dump(mode="json")
    d["item_id"] = d["item_id"][:-1] + ("0" if d["item_id"][-1] != "0" else "1")
    with pytest.raises(ValidationError, match="item_id"):
        Item.model_validate(d)


@pytest.mark.parametrize("kw,msg", [
    ({"condition": Condition.no_gold}, "cannot use condition"),
    ({"condition": Condition.gold_distractors}, "one gold passage"),
    ({"passages": []}, "exactly one gold"),
    ({"gold": {"answer": "x", "citations": ["Part 122"]}}, "bad citation"),
    ({"gold": {"answer": "x"}}, "answer and citations"),
])
def test_invalid_items(kw, msg):
    with pytest.raises(ValidationError, match=msg):
        build_item(**_qa(**kw))


def test_distractor_condition():
    it = build_item(**_qa(condition=Condition.gold_distractors,
                          passages=[_passage("distractor", "122.3", "P1"), _passage("gold", "122.2", "P2")]))
    assert [p.role for p in it.passages] == ["distractor", "gold"]


def test_abstention_requires_should_abstain():
    q = "What fee applies to an NPDES application?"
    base = dict(base_id=make_base_id(Task.abstention, "40-CFR-122.2", q, "0.1"), task=Task.abstention,
                condition=Condition.no_gold, question=q, as_of="2026-10-19",
                passages=[_passage("near_miss", "122.3")], split="test", lineage_id="40-CFR-122.2",
                title="40", part="122", provenance=_prov())
    with pytest.raises(ValidationError, match="should_abstain"):
        build_item(**base, gold={})
    assert build_item(**base, gold={"should_abstain": True}).gold.should_abstain


def test_superseded_outdated_only():
    q = "As of 2026-10-19, what must the quality manual contain?"
    label = {"outdated_section_id": "21-CFR-820.20@2019-01-01", "current_section_id": "21-CFR-820.10@2026-02-02",
             "change_type": "renumbered", "outdated_version_date": "2019-01-01",
             "current_version_date": "2026-02-02", "diff_id": "21-CFR-820.10@2026-02-02:renumbered"}
    base = dict(base_id=make_base_id(Task.superseded, "21-CFR-820.10", q, "0.1"), task=Task.superseded,
                condition=Condition.outdated_only, question=q, as_of="2026-10-19",
                passages=[{"passage_id": "P1", "section_id": "21-CFR-820.20@2019-01-01", "version_date": "2019-01-01",
                           "text": "old text", "role": "outdated"}],
                gold={"superseded": label, "should_abstain": True}, split="test", lineage_id="21-CFR-820.10",
                title="21", part="820", provenance=_prov(generator="superseded"))
    assert build_item(**base).gold.superseded.change_type == "renumbered"
