"""Stable identifiers.

section_id  = "{title}-CFR-{section}@{version_date}"   e.g. 40-CFR-122.26@2025-01-01
lineage_id  = "{title}-CFR-{section}"                   (date-free; follows renumbering via diffs)
citation    = "{title} CFR {section}{paragraph path}"   e.g. 40 CFR 122.26(b)(14)
"""
from __future__ import annotations

import hashlib
import re

_SECTION_ID = re.compile(r"^(?P<title>\d+)-CFR-(?P<section>[0-9A-Za-z.\-]+)@(?P<date>\d{4}-\d{2}-\d{2})$")
_CITATION = re.compile(
    r"^(?P<title>\d+)\s+CFR\s+(?:§\s*)?(?P<section>\d+[A-Za-z]?\.\d+[A-Za-z0-9\-]*)(?P<para>(?:\([A-Za-z0-9]+\))*)$"
)


def lineage_id(title: str, section: str) -> str:
    return f"{title}-CFR-{section}"


def section_id(title: str, section: str, version_date: str) -> str:
    return f"{lineage_id(title, section)}@{version_date}"


def parse_section_id(sid: str) -> dict:
    m = _SECTION_ID.match(sid)
    if not m:
        raise ValueError(f"bad section_id: {sid!r}")
    return m.groupdict()


def citation(title: str, section: str, label_path: str = "") -> str:
    return f"{title} CFR {section}{label_path}"


def parse_citation(text: str) -> dict | None:
    """Normalize a citation string; returns None if it is not a CFR section citation."""
    m = _CITATION.match(re.sub(r"\s+", " ", text.strip()))
    return m.groupdict() if m else None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))
