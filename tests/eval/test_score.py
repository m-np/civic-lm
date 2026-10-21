from civic_bench.schema import Condition, Task, build_item, make_base_id
from civic_eval.prompts import build_messages
from civic_eval.score import parse_output, score_item, sections_of


def _item(task=Task.grounded_qa, condition=Condition.gold_only, gold=None, passages=None):
    q = "What does Operator mean?"
    passages = passages if passages is not None else [
        {"passage_id": "P1", "section_id": "40-CFR-122.2@2019-02-12", "citation": "40 CFR 122.2",
         "version_date": "2019-02-12", "heading": "§ 122.2 Definitions.", "text": "Operator means x.", "role": "gold"}]
    return build_item(base_id=make_base_id(task, "40-CFR-122.2", q, "t"), task=task, condition=condition,
                      question=q, as_of="2026-10-19", passages=passages,
                      gold=gold or {"answer": "the person who runs the facility day to day",
                                    "citations": ["40 CFR 122.2(b)"], "acceptable_citations": ["40 CFR 122.2"]},
                      split="test", lineage_id="40-CFR-122.2", title="40", part="122",
                      provenance={"generator": "t", "generator_version": "t", "seed": 0,
                                  "question_source": "template", "dataset_version": "t"})


def test_parse_output_tolerates_fences_and_prose():
    out = parse_output('Sure!\n```json\n{"answer": "x", "citations": "40 CFR 122.2", "abstain": false}\n```')
    assert out.answer == "x" and out.citations == ["40 CFR 122.2"] and not out.abstain
    assert parse_output("no json here") is None


def test_sections_of_normalizes():
    assert sections_of(["40 CFR § 122.26(b)(14)", "40 C.F.R. 122.2", "see 40 CFR 122.26"]) == [
        "40 CFR 122.26", "40 CFR 122.2"]


def test_correct_answer_with_paragraph_citation():
    it = _item()
    out = parse_output('{"answer": "The person who runs the facility day to day.", '
                       '"citations": ["40 CFR 122.2(b)"], "abstain": false}')
    r = score_item(it, out, {"40 CFR 122.2"}, {"40:122"})
    assert r["correct"] and r["cit_p"] == 1.0 and r["cit_r"] == 1.0 and r["hallucinated"] == 0.0


def test_wrong_citation_and_hallucination():
    out = parse_output('{"answer": "The person who runs the facility day to day.", '
                       '"citations": ["40 CFR 122.99"], "abstain": false}')
    r = score_item(_item(), out, {"40 CFR 122.2"}, {"40:122"})
    assert not r["correct"] and r["cit_p"] == 0.0 and r["hallucinated"] == 1.0


def test_abstention_scoring():
    it = _item(task=Task.abstention, condition=Condition.no_gold, gold={"should_abstain": True},
               passages=[{"passage_id": "P1", "section_id": "40-CFR-122.3@2019-02-12", "version_date": "2019-02-12",
                          "text": "unrelated", "role": "near_miss"}])
    assert score_item(it, parse_output('{"abstain": true}'), set(), set())["correct"]
    assert not score_item(it, parse_output('{"answer": "guess", "abstain": false}'), set(), set())["correct"]
    assert not score_item(it, None, set(), set())["format_ok"]


def test_prompts_render_validity_and_tiers():
    it = _item()
    naive = build_messages(it, "naive", {"40-CFR-122.2@2019-02-12": {"from": "2019-02-12", "to": None}})
    eng = build_messages(it, "engineered", {"40-CFR-122.2@2019-02-12": {"from": "2019-02-12", "to": "2025-01-01"}})
    assert len(naive) == 1 and "no end date recorded" in naive[0]["content"]
    assert eng[0]["role"] == "system" and "until 2025-01-01" in eng[-1]["content"]
    assert "rule in force on 2026-10-19" in eng[-1]["content"]
