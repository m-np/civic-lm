import pytest
from civic_data.config import DEFAULT_CONFIG, load_scope
from civic_data.ids import citation, parse_citation, parse_section_id, section_id
from civic_data.ingest.ecfr import history_fetch_list, section_versions


def test_section_id_roundtrip():
    sid = section_id("40", "122.26", "2025-01-01")
    assert sid == "40-CFR-122.26@2025-01-01"
    assert parse_section_id(sid) == {"title": "40", "section": "122.26", "date": "2025-01-01"}


@pytest.mark.parametrize("text,ok", [("40 CFR 122.26(b)(14)", True), ("40 CFR § 122.26", True),
                                     ("40  CFR 122.26", True), ("Part 122", False), ("40 CFR 122", False)])
def test_parse_citation(text, ok):
    assert (parse_citation(text) is not None) == ok
    assert citation("40", "122.26", "(b)(14)") == "40 CFR 122.26(b)(14)"


def test_scope_parts_are_disjoint_and_dates_ordered():
    scope = load_scope(DEFAULT_CONFIG)
    parts = scope.parts()
    assert len({(t, p) for t, p, _ in parts}) == len(parts)
    assert {s for *_, s in parts} == {"train", "dev", "test"}
    assert scope.history_start < scope.test_cutoff < scope.as_of
    assert scope.snapshot_dates == sorted(scope.snapshot_dates)


def test_section_versions_merge_reissues(fixtures):
    import json
    vj = json.loads((fixtures / "versions_title-40_part-122.json").read_text())
    rows = section_versions(vj)
    keys = [(r["identifier"], r["date"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(r["date"] for r in rows)
    scope = load_scope(DEFAULT_CONFIG)
    for _sec, date in history_fetch_list(scope, vj):
        assert scope.history_start < date <= scope.as_of
