"""Stage 2: eCFR XML -> data/processed/sections.parquet + provenance.parquet.

One row per (section, snapshot date). Each row keeps the section's paragraphs with their
label paths, e.g. "(b)(14)(ii)", so generators can cite below the section level.

Every parsed section passes a parse-vs-raw check: the normalized paragraph text, with all
whitespace removed, must equal the raw XML text of the same elements. A failure is a parser
bug and stops the build (see "Parser errors" in the plan).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from lxml import etree

from .config import DEFAULT_CONFIG, Scope, load_scope
from .ids import lineage_id, section_id, sha256_bytes, sha256_text
from .ingest.ecfr import part_rel, section_versions, versions_rel

log = logging.getLogger(__name__)

# Children of a section DIV8 that are metadata, not regulatory text.
# XREF: "Link to an amendment published at ..." pointers; EDNOTE: editorial notes.
META_TAGS = {"HEAD", "CITA", "SECAUTH", "XREF", "EDNOTE"}
PARA_TAGS = {"P", "FP"}

_WS = re.compile(r"\s+")
# Paragraph label alphabet: (a) (aa) (12) (iv) (A) (AA). Abbreviations such as "(DHF)" are not labels.
_LABEL = r"(?:[a-z]{1,2}|[ivxlc]{1,7}|[A-Z]{1,2}|\d{1,3})"
_LEAD_LABELS = re.compile(rf"^\s*((?:\((?:<I>)?{_LABEL}(?:</I>)?\)\s*)+)")
_ONE_LABEL = re.compile(rf"\((<I>)?({_LABEL})(</I>)?\)")
# An italic paragraph heading ("<I>Duty to apply.</I>") followed directly by another label.
# A short paragraph heading followed directly by another label: italic ("<I>Duty to apply.</I> (1)")
# or plain text ending in "." / "—" / ":" ("Bypass—(1) Definitions. (i) ...").
_HEADING_THEN_LABEL = re.compile(
    rf"^(\s*(?:<I>[^<]{{2,200}}</I>\s*[.—:]?|[^()<]{{1,80}}?[.—:])\s*)\((?:<I>)?{_LABEL}(?:</I>)?\)")
_ROMAN = re.compile(r"^(?=[ivxlcdm]+$)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
# "<I>Term</I> means", also after labels "(a) <I>Act</I> means" or in quotes "“<I>Batch</I>” means".
_DEFINED_TERM = re.compile(
    r"^\s*(?:\((?:<I>)?\w{1,6}(?:</I>)?\)\s*)*[“\"]?\s*<I>([^<]{1,120})</I>\s*[”\"]?\s*(?:\([^)]*\)\s*)?"
    r"(?:means|is defined|includes|has the meaning|refers to)")


class ParseCheckError(ValueError):
    pass


@dataclass
class Paragraph:
    label_path: str          # "(b)(14)(ii)"; "" for unlabeled text before the first label
    labeled: bool            # False for continuation text that inherits the current path
    kind: str                # "p" for P/FP, else the lowercased tag (table, note, extract, ...)
    text: str
    defined_term: str | None = None


@dataclass
class SectionRecord:
    section_id: str
    lineage_id: str
    title: str
    part: str
    subpart: str | None
    section: str
    version_date: str
    snapshot_date: str
    heading: str
    reserved: bool
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    citation_note: str | None = None
    text_hash: str = ""
    source_path: str = ""
    source_sha256: str = ""


# --------------------------------------------------------------------------- text helpers

def collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def no_ws(s: str) -> str:
    return _WS.sub("", s)


def element_text(el: etree._Element) -> str:
    """Readable text of an element. Table cells are tab-separated, rows newline-separated,
    so the parse-vs-raw check (which strips whitespace) still matches exactly."""
    if el.tag in ("GPOTABLE", "TABLE"):
        rows = []
        for row in el.iter("ROW", "TR", "BOXHD", "TTITLE", "CAPTION"):
            cells = [collapse("".join(c.itertext())) for c in row if isinstance(c.tag, str)]
            if row.tag in ("TTITLE", "CAPTION") or not cells:
                cells = [collapse("".join(row.itertext()))]
            rows.append("\t".join(cells))
        text = "\n".join(r for r in rows if r)
        # Text outside rows (rare: stray notes inside the table) is kept, not dropped.
        if no_ws(text) != no_ws("".join(el.itertext())):
            text = collapse("".join(el.itertext()))
        return text
    return collapse("".join(el.itertext()))


def inner_xml(el: etree._Element) -> str:
    s = etree.tostring(el, encoding="unicode", with_tail=False)
    return s[s.index(">") + 1: s.rindex("<")] if s.endswith(">") and "</" in s else ""


# --------------------------------------------------------------------------- paragraph labels

def _successor(label: str) -> str:
    """Next letter label: a -> b, z -> aa, aa -> bb (CFR doubles letters after z)."""
    if label in ("z", "Z"):
        return label * 2
    if len(label) == 2 and label[0] == label[1]:
        return chr(ord(label[0]) + 1) * 2
    return chr(ord(label) + 1) if len(label) == 1 else ""


def _roman_to_int(s: str) -> int:
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    for a, b in zip(s, s[1:] + " ", strict=False):
        total += -vals[a] if b != " " and vals[a] < vals[b] else vals[a]
    return total


class LabelStack:
    """Tracks the CFR paragraph hierarchy: (a) -> (1) -> (i) -> (A) -> (<I>1</I>) -> (<I>i</I>).

    (i), (v), (x) etc. are ambiguous between lowercase-letter level 1 and roman level 3.
    A label is a valid continuation at a level if it is the first label at that level
    or the successor of the label currently held there. When both readings are valid,
    the next label decides: "(h)(8)" then "(i)" then "(1)" or "(j)" means (i) is the letter;
    then "(ii)" or "(A)" means it is roman. With no usable next label, roman wins.
    """

    def __init__(self) -> None:
        self.levels: dict[int, str] = {}

    def path(self) -> str:
        return "".join(f"({self.levels[k]})" for k in sorted(self.levels))

    def _fits(self, level: int, label: str) -> bool:
        if level > 1 and level - 1 not in self.levels:
            return False  # CFR nesting never skips a level: (a)(1)(i), not (a)(i)
        cur = self.levels.get(level)
        if level == 1:
            return label == "a" if cur is None else label == _successor(cur)
        if level in (2, 5):
            return label == "1" if cur is None else label.isdigit() and int(label) == int(cur) + 1
        if level in (3, 6):
            if not _ROMAN.match(label):
                return False
            return label == "i" if cur is None else _roman_to_int(label) == _roman_to_int(cur) + 1
        if level == 4:
            return label == "A" if cur is None else label == _successor(cur)
        return False

    def _candidates(self, label: str, italic: bool) -> list[int]:
        if label.isdigit():
            return [5] if italic else [2]
        if label.isupper():
            return [4]
        roman = bool(_ROMAN.match(label))
        if italic:
            return [6] if roman else []
        if roman and len(label) == 1:
            return [3, 1]
        if roman:
            return [3]
        return [1] if len(label) == 1 or label[0] == label[1] else []

    def push(self, label: str, italic: bool, next_label: str | None = None) -> bool:
        cands = self._candidates(label, italic)
        if not cands:
            return False
        fitting = [lv for lv in cands if self._fits(lv, label)]
        if fitting == [3, 1] and next_label is not None and _next_says_letter(label, next_label):
            fitting = [1]
        # A label that fits nowhere (gaps in the source, "[Reserved]" jumps) still takes
        # its first candidate level, so a numbering gap does not misplace later labels.
        level = fitting[0] if fitting else cands[0]
        if level > 1 and not any(k < level for k in self.levels) and not fitting:
            return False
        self.levels = {k: v for k, v in self.levels.items() if k < level}
        self.levels[level] = label
        return True


def _next_says_letter(label: str, nxt: str) -> bool:
    """For an ambiguous (i)/(v)/(x)...: does the following label imply the letter reading?"""
    if nxt == "1" or nxt == _successor(label):
        return True
    return False


def split_paragraph_labels(p_inner_xml: str) -> list[tuple[str, bool]]:
    """Leading labels of a paragraph: '(b)(14)(<I>ii</I>) Text' -> [('b',F),('14',F),('ii',T)].

    Labels may continue after an italic paragraph heading:
    '(a) <I>Duty to apply.</I> (1) Any person...' -> [('a',F),('1',F)].
    """
    out: list[tuple[str, bool]] = []
    rest = p_inner_xml
    while True:
        m = _LEAD_LABELS.match(rest)
        if not m:
            return out
        out.extend((g[1], bool(g[0])) for g in _ONE_LABEL.findall(m.group(1)))
        rest = rest[m.end():]
        h = _HEADING_THEN_LABEL.match(rest)
        if not h:
            return out
        rest = rest[h.end(1):]


# --------------------------------------------------------------------------- section parsing

def parse_section(div: etree._Element, *, title: str, version_date: str, snapshot_date: str,
                  subpart: str | None, source_path: str = "", source_sha256: str = "") -> SectionRecord:
    number = div.get("N", "").strip()
    head_el = div.find("HEAD")
    heading = collapse("".join(head_el.itertext())) if head_el is not None else ""
    stack = LabelStack()
    paragraphs: list[Paragraph] = []
    citation_note = None
    raw_body = []
    # Leading labels of every paragraph, flattened, so each label can see the next one.
    children = [c for c in div if isinstance(c.tag, str)]
    labels_of = {id(c): split_paragraph_labels(inner_xml(c)) for c in children if c.tag in PARA_TAGS}
    flat = [lab for c in children for lab, _ in labels_of.get(id(c), [])]
    pos = 0
    for child in children:
        if not isinstance(child.tag, str):
            continue
        if child.tag == "CITA":
            citation_note = collapse("".join(child.itertext()))
        if child.tag in META_TAGS:
            continue
        raw_body.append("".join(child.itertext()))
        text = element_text(child)
        if child.tag in PARA_TAGS:
            inner = inner_xml(child)
            labels = labels_of[id(child)]
            pushed = []
            for lab, it in labels:
                pos += 1
                pushed.append(stack.push(lab, it, flat[pos] if pos < len(flat) else None))
            term = _DEFINED_TERM.match(inner)
            paragraphs.append(Paragraph(
                label_path=stack.path(), labeled=bool(labels) and all(pushed), kind="p",
                text=text, defined_term=collapse(term.group(1)) if term else None))
        else:
            paragraphs.append(Paragraph(label_path=stack.path(), labeled=False,
                                        kind=child.tag.lower(), text=text))
    body = "\n".join(p.text for p in paragraphs if p.text)
    if no_ws(body) != no_ws("".join(raw_body)):
        raise ParseCheckError(f"{title} CFR {number} @{snapshot_date}: parsed text differs from raw XML")
    return SectionRecord(
        section_id=section_id(title, number, version_date),
        lineage_id=lineage_id(title, number),
        title=title, part=number.split(".")[0], subpart=subpart, section=number,
        version_date=version_date, snapshot_date=snapshot_date, heading=heading,
        reserved="[Reserved]" in heading or "[Removed]" in heading,
        text=body, paragraphs=[p for p in paragraphs if p.text], citation_note=citation_note,
        text_hash=sha256_text(no_ws(body)), source_path=source_path, source_sha256=source_sha256,
    )


def section_divs(root: etree._Element) -> list[etree._Element]:
    if root.get("TYPE") == "SECTION":
        return [root]
    return [d for d in root.iter() if isinstance(d.tag, str) and d.get("TYPE") == "SECTION"]


def subpart_of(div: etree._Element) -> str | None:
    for anc in div.iterancestors():
        if anc.get("TYPE") == "SUBPART":
            return anc.get("N")
    return None


def load_xml(path: Path) -> etree._Element:
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False, no_network=True,
                             huge_tree=True)
    return etree.fromstring(path.read_bytes(), parser)


class VersionIndex:
    """Effective version date of a section on a given snapshot date, from the versioner API."""

    def __init__(self, versions_json: dict):
        self.by_section: dict[str, list[dict]] = {}
        for v in section_versions(versions_json):
            self.by_section.setdefault(v["identifier"], []).append(v)

    def version_on(self, section: str, date: str) -> dict | None:
        best = None
        for v in self.by_section.get(section, []):
            if v["date"] <= date:
                best = v
        return best

    def subpart(self, section: str) -> str | None:
        vs = self.by_section.get(section)
        return vs[-1]["subpart"] if vs else None


def parse_file(path: Path, *, raw_dir: Path, title: str, snapshot_date: str,
               index: VersionIndex) -> list[SectionRecord]:
    root = load_xml(path)
    sha = sha256_bytes(path.read_bytes())
    rel = str(path.relative_to(raw_dir))
    out = []
    for div in section_divs(root):
        number = div.get("N", "").strip()
        v = index.version_on(number, snapshot_date)
        out.append(parse_section(
            div, title=title, snapshot_date=snapshot_date,
            version_date=v["date"] if v else snapshot_date,
            subpart=subpart_of(div) or index.subpart(number),
            source_path=rel, source_sha256=sha))
    return out


# --------------------------------------------------------------------------- tables

def records_to_frame(records: list[SectionRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = asdict(r)
        d["paragraphs"] = json.dumps(d["paragraphs"], ensure_ascii=False)
        rows.append(d)
    cols = list(SectionRecord.__dataclass_fields__)
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["title", "part", "lineage_id", "snapshot_date"], kind="stable").reset_index(drop=True)


def paragraphs_of(row) -> list[dict]:
    return json.loads(row["paragraphs"])


def build_sections(scope: Scope) -> pd.DataFrame:
    raw = scope.paths.raw
    records: list[SectionRecord] = []
    for title, part, _split in scope.parts():
        vpath = raw / versions_rel(title, part)
        if not vpath.exists():
            log.warning("missing %s; run ingest first", vpath)
            continue
        index = VersionIndex(json.loads(vpath.read_text()))
        for date in scope.snapshot_dates:
            p = raw / part_rel(title, part, date)
            if p.exists():
                recs = parse_file(p, raw_dir=raw, title=title, snapshot_date=date, index=index)
                records.extend(recs)
                log.info("%s CFR %s @%s: %d sections", title, part, date, len(recs))
    df = records_to_frame(records)
    df.insert(4, "split", [scope.split_of(t, p) for t, p in zip(df["title"], df["part"], strict=False)])
    return df


def build_provenance(scope: Scope) -> pd.DataFrame:
    log_path = scope.paths.raw / "_provenance.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()] if log_path.exists() else []
    df = pd.DataFrame(rows, columns=["path", "source_url", "license", "retrieved_at", "sha256", "bytes"])
    # A path is written once; keep the first record if a log line was ever duplicated.
    return df.drop_duplicates("path", keep="first").sort_values("path").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    from .manifest import write_tables

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scope = load_scope(args.config, args.data_dir)
    write_tables(scope, {"sections": build_sections(scope), "provenance": build_provenance(scope)})


if __name__ == "__main__":
    main()
