from civic_data.diff_classify import Version, classify_lineage, pair_renumberings, paragraph_changes
from civic_data.ids import section_id, sha256_text


def V(sec, date, text, paras=None, **kw):
    paras = paras if paras is not None else [{"label_path": "(a)", "text": text}]
    return Version(lineage_id=f"40-CFR-{sec}", section_id=section_id("40", sec, date), title="40",
                   part=sec.split(".")[0], section=sec, version_date=date, text=text,
                   text_hash=sha256_text(text) if text else "", paragraphs=paras, **kw)


def types(diffs):
    return [d.change_type for d in diffs]


def test_amended_with_paragraph_detail():
    old = V("122.5", "2020-01-01", "x", [{"label_path": "(a)", "text": "keep"}, {"label_path": "(b)", "text": "old"}])
    new = V("122.5", "2024-03-01", "y", [{"label_path": "(a)", "text": "keep"}, {"label_path": "(b)", "text": "new"},
                                          {"label_path": "(c)", "text": "added"}])
    (d,) = classify_lineage([old, new], history_start="2023-01-01", part_first_date="2017-01-01")
    assert d.change_type == "amended"
    assert d.from_section_id == "40-CFR-122.5@2020-01-01" and d.to_section_id == "40-CFR-122.5@2024-03-01"
    assert d.changed_paragraphs == [{"label_path": "(b)", "change": "modified"},
                                    {"label_path": "(c)", "change": "added"}]


def test_same_text_is_no_diff_and_editorial_is_separate():
    a = V("1.1", "2020-01-01", "same")
    b = V("1.1", "2024-01-01", "same")
    c = V("1.1", "2024-06-01", "same, fixed typo", substantive=False)
    assert types(classify_lineage([a, b, c], history_start="2023-01-01", part_first_date="2017-01-01")) == ["editorial"]


def test_changes_before_history_start_are_ignored():
    a, b = V("1.1", "2018-01-01", "a"), V("1.1", "2019-01-01", "b")
    assert classify_lineage([a, b], history_start="2023-01-01", part_first_date="2017-01-01") == []


def test_added_removed_and_baseline():
    new_sec = [V("1.7", "2024-05-01", "brand new")]
    assert types(classify_lineage(new_sec, history_start="2023-01-01", part_first_date="2017-01-01")) == ["added"]
    baseline = [V("1.8", "2017-01-01", "old")]
    assert classify_lineage(baseline, history_start="2023-01-01", part_first_date="2017-01-01") == []
    gone = [V("1.9", "2020-01-01", "t"), V("1.9", "2024-01-01", "", paras=[], removed=True)]
    (d,) = classify_lineage(gone, history_start="2023-01-01", part_first_date="2017-01-01")
    assert d.change_type == "removed" and d.to_section_id is None


def test_renumbering_pairs_removed_and_added_same_day():
    body = "Each owner or operator shall submit the report within 30 days " * 5
    removed = classify_lineage([V("60.10", "2020-01-01", body), V("60.10", "2025-02-01", "", paras=[], removed=True)],
                               history_start="2023-01-01", part_first_date="2017-01-01")
    added = classify_lineage([V("60.12", "2025-02-01", body + " promptly")],
                             history_start="2023-01-01", part_first_date="2017-01-01")
    text_of = {"40-CFR-60.10@2020-01-01": body, "40-CFR-60.12@2025-02-01": body + " promptly"}
    (d,) = pair_renumberings(removed + added, text_of)
    assert d.change_type == "renumbered"
    assert d.from_lineage_id == "40-CFR-60.10" and d.lineage_id == "40-CFR-60.12"


def test_unrelated_same_day_changes_stay_separate():
    removed = classify_lineage([V("60.10", "2020-01-01", "alpha beta gamma"),
                                V("60.10", "2025-02-01", "", paras=[], removed=True)],
                               history_start="2023-01-01", part_first_date="2017-01-01")
    added = classify_lineage([V("60.12", "2025-02-01", "completely different words here")],
                             history_start="2023-01-01", part_first_date="2017-01-01")
    out = pair_renumberings(removed + added, {"40-CFR-60.10@2020-01-01": "alpha beta gamma",
                                              "40-CFR-60.12@2025-02-01": "completely different words here"})
    assert sorted(types(out)) == ["added", "removed"]


def test_paragraph_changes_removed():
    assert paragraph_changes([{"label_path": "(a)", "text": "x"}], []) == [{"label_path": "(a)", "change": "removed"}]
