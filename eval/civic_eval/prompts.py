"""Prompt tiers. Every model gets identical passages; only the instructions differ by tier.

naive       : one-line instruction plus the JSON format.
engineered  : rules for citing, abstaining and checking version dates, plus few-shot examples.
              The examples use a fictional "Title 99" so no benchmark text leaks into them.
"""
from __future__ import annotations

from civic_bench.schema import Item

PROMPT_VERSION = "0.2"  # 0.2: closed-book items get their own instructions
TIERS = ("naive", "engineered")

JSON_FORMAT = ('{"answer": string or null, "citations": [list of "<title> CFR <section>" strings], '
               '"abstain": true or false, "outdated_warning": true or false}')

ENGINEERED_SYSTEM = f"""You answer questions about U.S. federal regulations using only the passages provided.

Rules:
1. Use only the supplied passages. Do not use outside knowledge.
2. Cite every section you rely on as "<title> CFR <section>" (for example "99 CFR 10.5" or "99 CFR 10.5(b)(2)"). Cite only passages you used.
3. If the passages do not contain the answer, set "abstain": true, "answer": null, "citations": [].
4. Each passage shows the period it was in force. The question asks about the rule in force on a specific date. If a passage was no longer in force on that date, do not present it as the current rule: set "outdated_warning": true. If a passage in force on that date answers the question, answer from it. If none does, also set "abstain": true.
5. Reply with one JSON object only, in this format: {JSON_FORMAT}"""

FEW_SHOT = [
    ("""Passages:
[P1] 99 CFR 10.2 — Definitions. (in force from 2019-04-01, no end date recorded)
Facility means any building or site used to store regulated materials.
Operator means the person responsible for day-to-day operation of a facility.

Question (rule in force on 2026-10-19): Under the fictional storage rules, what does "Operator" mean?""",
     '{"answer": "The person responsible for day-to-day operation of a facility.", "citations": ["99 CFR 10.2"], '
     '"abstain": false, "outdated_warning": false}'),
    ("""Passages:
[P1] 99 CFR 10.4 — Inspections. (in force from 2019-04-01, no end date recorded)
(a) Each facility must be inspected at least once a year.

Question (rule in force on 2026-10-19): Under the fictional storage rules, what fee is charged for an inspection?""",
     '{"answer": null, "citations": [], "abstain": true, "outdated_warning": false}'),
    ("""Passages:
[P1] 99 CFR 10.6 — Reports. (in force from 2018-01-01 until 2025-03-01)
(a) Operators must file a report within 30 days of an incident.
[P2] 99 CFR 10.6 — Reports. (in force from 2025-03-01, no end date recorded)
(a) Operators must file a report within 10 days of an incident.

Question (rule in force on 2026-10-19): Under the fictional storage rules, how soon must an incident report be filed?""",
     '{"answer": "Within 10 days of an incident.", "citations": ["99 CFR 10.6(a)"], '
     '"abstain": false, "outdated_warning": true}'),
]


CLOSED_BOOK_SYSTEM = f"""You answer questions about U.S. federal regulations from your own knowledge. No passages are provided.

Rules:
1. Answer if you know the rule. Cite the section you rely on as "<title> CFR <section>". Do not invent section numbers.
2. If you do not know, set "abstain": true, "answer": null, "citations": [].
3. Reply with one JSON object only, in this format: {JSON_FORMAT}"""


def render_passages(item: Item, validity: dict) -> str:
    lines = []
    for p in item.passages:
        v = validity.get(p.section_id or "", {})
        start = v.get("from") or p.version_date
        period = (f"in force from {start} until {v['to']}" if v.get("to")
                  else f"in force from {start}, no end date recorded")
        cite = p.citation or p.doc_id
        lines.append(f"[{p.passage_id}] {cite} — {p.heading.split(' ', 2)[-1] if p.heading else ''} ({period})\n{p.text}")
    return "\n".join(lines)


def prompt_sha(messages: list[dict]) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def build_messages(item: Item, tier: str, validity: dict) -> list[dict]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    q = f"Question (rule in force on {item.as_of}): {item.question}"
    if not item.passages:  # closed-book control: same instructions in both tiers
        return [{"role": "system", "content": CLOSED_BOOK_SYSTEM}, {"role": "user", "content": q}]
    body = f"Passages:\n{render_passages(item, validity)}\n\n{q}"
    if tier == "naive":
        return [{"role": "user", "content":
                 f"{body}\n\nAnswer using the passages and reply with JSON only: {JSON_FORMAT}"}]
    msgs = [{"role": "system", "content": ENGINEERED_SYSTEM}]
    for u, a in FEW_SHOT:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": body})
    return msgs
