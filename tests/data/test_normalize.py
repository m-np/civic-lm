import json

import pytest
from civic_data.normalize import (
    LabelStack,
    ParseCheckError,
    VersionIndex,
    parse_file,
    parse_section,
    split_paragraph_labels,
)
from lxml import etree


def _parse(fixtures):
    index = VersionIndex(json.loads((fixtures / "versions_title-40_part-122.json").read_text()))
    return {r.section: r for r in parse_file(fixtures / "title-40_part-122_2025-01-01.xml",
                                             raw_dir=fixtures, title="40",
                                             snapshot_date="2025-01-01", index=index)}


def test_real_part_parses_and_passes_raw_check(fixtures):
    recs = _parse(fixtures)
    assert set(recs) == {"122.1", "122.2", "122.21"}
    r = recs["122.2"]
    assert r.heading.startswith("§ 122.2 Definitions")
    assert r.subpart == "A"
    assert r.lineage_id == "40-CFR-122.2"
    # version date comes from the versioner API, not the snapshot date
    assert r.section_id == f"40-CFR-122.2@{r.version_date}" and r.version_date <= "2025-01-01"
    assert r.citation_note and r.citation_note.startswith("[48 FR 14153")


def test_definitions_are_captured(fixtures):
    terms = {p.defined_term for p in _parse(fixtures)["122.2"].paragraphs if p.defined_term}
    assert {"Administrator", "Best management practices", "Whole effluent toxicity"} <= terms


def test_label_paths_in_real_section(fixtures):
    paths = [p.label_path for p in _parse(fixtures)["122.21"].paragraphs if p.labeled]
    # "(a) <I>Duty to apply.</I> (1) Any person..." is paragraph (a)(1)
    assert paths[:3] == ["(a)(1)", "(a)(2)(i)", "(a)(2)(i)(A)"]
    assert "(a)(2)(ii)" in paths
    assert any(p.count("(") >= 4 for p in paths), "expected deep paths such as (g)(7)(i)(A)"
    assert len(paths) == len(set(paths)), "labeled paths must be unique within a section"


@pytest.mark.parametrize("seq,expected", [
    (["a", "1", "i", "ii", "2", "b"], ["(a)", "(a)(1)", "(a)(1)(i)", "(a)(1)(ii)", "(a)(2)", "(b)"]),
    # (i) right after (h) with no level-2 parent is the letter
    (["h", "i"], ["(h)", "(i)"]),
    # (i) after (h)(3) is the roman subparagraph
    (["h", "3", "i"], ["(h)", "(h)(3)", "(h)(3)(i)"]),
    # (v) after (iv) is roman, (v) after (u) is a letter
    (["a", "1", "iv", "v"], ["(a)", "(a)(1)", "(a)(1)(iv)", "(a)(1)(v)"]),
    (["u", "v"], ["(u)", "(v)"]),
    (["a", "1", "i", "A", "B"], ["(a)", "(a)(1)", "(a)(1)(i)", "(a)(1)(i)(A)", "(a)(1)(i)(B)"]),
])
def test_label_stack(seq, expected):
    s, got = LabelStack(), []
    for lab in seq:
        assert s.push(lab, italic=False)
        got.append(s.path())
    assert got == expected


def test_italic_levels():
    s = LabelStack()
    for lab in ["a", "1", "i", "A"]:
        s.push(lab, False)
    s.push("1", True)
    s.push("i", True)
    assert s.path() == "(a)(1)(i)(A)(1)(i)"


def test_split_leading_labels():
    assert split_paragraph_labels("(b)(14)(<I>ii</I>) Text") == [("b", False), ("14", False), ("ii", True)]
    assert split_paragraph_labels("<I>Term</I> means x") == []


def _div(body: str) -> etree._Element:
    return etree.fromstring(f'<DIV8 N="1.5" TYPE="SECTION"><HEAD>§ 1.5 Test.</HEAD>{body}</DIV8>')


def test_table_text_kept_and_checked():
    div = _div("<P>(a) See table.</P><GPOTABLE><BOXHD><CHED>Col A</CHED><CHED>Col B</CHED></BOXHD>"
               "<ROW><ENT>1</ENT><ENT>2</ENT></ROW></GPOTABLE><CITA>[1 FR 1]</CITA>")
    r = parse_section(div, title="1", version_date="2020-01-01", snapshot_date="2020-01-01", subpart=None)
    assert [p.kind for p in r.paragraphs] == ["p", "gpotable"]
    assert "Col A\tCol B" in r.text and "1\t2" in r.text
    assert r.paragraphs[1].label_path == "(a)"
    assert r.citation_note == "[1 FR 1]" and "1 FR 1" not in r.text


def test_reserved_section():
    div = etree.fromstring('<DIV8 N="1.9" TYPE="SECTION"><HEAD>§ 1.9 [Reserved]</HEAD></DIV8>')
    r = parse_section(div, title="1", version_date="2020-01-01", snapshot_date="2020-01-01", subpart=None)
    assert r.reserved and r.text == ""


def test_parse_check_catches_dropped_text(monkeypatch):
    import civic_data.normalize as n
    monkeypatch.setattr(n, "element_text", lambda el: "")
    with pytest.raises(ParseCheckError):
        parse_section(_div("<P>(a) Text.</P>"), title="1", version_date="2020-01-01",
                      snapshot_date="2020-01-01", subpart=None)


def test_labels_after_italic_heading():
    assert split_paragraph_labels("(a) <I>Duty to apply.</I> (1) Any person") == [("a", False), ("1", False)]
    assert split_paragraph_labels("(2) <I>Application Forms:</I> (i) All applicants") == [("2", False), ("i", False)]
    # italic heading with no label after it stays a heading
    assert split_paragraph_labels("(e) <I>Electronic reporting.</I> If documents") == [("e", False)]


def test_lookahead_resolves_letter_i():
    s = LabelStack()
    for lab in ["h", "8"]:
        s.push(lab, False)
    s.push("i", False, next_label="1")
    assert s.path() == "(i)"
    s2 = LabelStack()
    for lab in ["h", "8"]:
        s2.push(lab, False)
    s2.push("i", False, next_label="ii")
    assert s2.path() == "(h)(8)(i)"


def test_abbreviation_is_not_a_label():
    assert split_paragraph_labels("(e) <I>Design history file</I> (DHF) means") == [("e", False)]
    assert split_paragraph_labels("(aa) <I>Term</I> means") == [("aa", False)]


def test_amendment_link_xref_excluded():
    div = _div('<XREF ID="1">Link to an amendment published at 89 FR 82945.</XREF><P>(a) Text.</P>')
    r = parse_section(div, title="1", version_date="2020-01-01", snapshot_date="2020-01-01", subpart=None)
    assert r.text == "(a) Text." and r.paragraphs[0].label_path == "(a)"


def test_double_letter_labels():
    s = LabelStack()
    for lab in ["y", "z", "aa", "bb"]:
        assert s.push(lab, False)
    assert s.path() == "(bb)"


def test_defined_term_forms():
    from civic_data.normalize import _DEFINED_TERM
    for xml, term in [("<I>Act</I> means x", "Act"), ("(a) <I>Act</I> means x", "Act"),
                      ("(e) <I>Design history file</I> (DHF) means x", "Design history file"),
                      ("“<I>Batch or lot</I>” means x", "Batch or lot")]:
        assert _DEFINED_TERM.match(xml).group(1) == term
    assert _DEFINED_TERM.match("(a) The Administrator shall") is None


def test_labels_after_plain_heading():
    assert split_paragraph_labels("(m) Bypass—(1) Definitions. (i) Bypass means") == [
        ("m", False), ("1", False), ("i", False)]
    # a reference inside running text is not a label
    assert split_paragraph_labels("(b) The permittee must comply with paragraph (a) of this section") == [("b", False)]


def test_labels_after_italic_heading_and_dash():
    assert split_paragraph_labels("(m) <I>Bypass</I>—(1) <I>Definitions.</I> (i) <I>Bypass</I> means") == [
        ("m", False), ("1", False), ("i", False)]
