"""Structural validation of the evaluation dataset.

Runs before any metric does. A label whose anchor no longer appears in the
corpus is worse than a missing label: it silently scores against nothing while
the eval still reports a number. This catches that, and it has to keep passing
every time the corpus or the chunking changes.

Standalone:  python -m tests.eval.validate_dataset
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tests.eval.anchors import DOCS_DIR, anchor_in_document

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

REQUIRED_FIELDS = {
    "id",
    "question",
    "ground_truth",
    "source_docs",
    "source_anchors",
    "expected_behavior",
    "type",
}
VALID_BEHAVIORS = {"answer", "refuse"}
VALID_TYPES = {"factual", "multi_hop", "unanswerable", "ambiguous"}

# Anchors shorter than this match too much text to be evidence of retrieval.
MIN_ANCHOR_CHARS = 25


def load_dataset() -> list[dict]:
    """Every record across every batch file, in id order."""
    records: list[dict] = []
    for path in sorted(DATASET_DIR.glob("batch*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno} is not valid JSON: {exc}") from exc
    return sorted(records, key=lambda r: r["id"])


def validate(records: list[dict]) -> list[str]:
    """Return a list of problems. Empty means the dataset is structurally sound."""
    problems: list[str] = []
    known_docs = {p.name for p in DOCS_DIR.glob("*.md")}

    for rid, count in Counter(r.get("id") for r in records).items():
        if count > 1:
            problems.append(f"duplicate id {rid} appears {count} times")

    seen_questions: dict[str, str] = {}

    for record in records:
        rid = record.get("id", "<missing id>")

        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            problems.append(f"{rid}: missing fields {sorted(missing)}")
            continue

        behavior = record["expected_behavior"]
        if behavior not in VALID_BEHAVIORS:
            problems.append(f"{rid}: expected_behavior {behavior!r} is not valid")
        if record["type"] not in VALID_TYPES:
            problems.append(f"{rid}: type {record['type']!r} is not valid")

        question = record["question"].strip().lower()
        if question in seen_questions:
            problems.append(f"{rid}: question duplicates {seen_questions[question]}")
        else:
            seen_questions[question] = rid

        docs = record["source_docs"]
        anchors = record["source_anchors"]
        if not isinstance(docs, list) or not isinstance(anchors, list):
            problems.append(f"{rid}: source_docs and source_anchors must be lists")
            continue

        # A refusal case is defined by having nothing in the corpus to cite.
        # Anything else must cite evidence, or context recall scores nothing.
        if behavior == "refuse":
            if docs or anchors:
                problems.append(f"{rid}: refuse case must cite no docs and no anchors")
            if record["type"] != "unanswerable":
                problems.append(f"{rid}: refuse case must be type unanswerable")
            continue

        if record["type"] == "unanswerable":
            problems.append(f"{rid}: type unanswerable must have expected_behavior refuse")
        if not docs:
            problems.append(f"{rid}: answer case cites no source_docs")
        if not anchors:
            problems.append(f"{rid}: answer case cites no source_anchors")

        for doc in docs:
            if doc not in known_docs:
                problems.append(f"{rid}: source doc {doc!r} is not in the corpus")

        for anchor in anchors:
            if len(anchor) < MIN_ANCHOR_CHARS:
                problems.append(
                    f"{rid}: anchor is only {len(anchor)} chars, "
                    f"minimum {MIN_ANCHOR_CHARS}: {anchor!r}"
                )
                continue
            if not any(d in known_docs for d in docs):
                continue
            if not anchor_in_document(anchor, docs):
                problems.append(f"{rid}: anchor not found in {docs}: {anchor!r}")

    return problems


def summarize(records: list[dict]) -> str:
    by_type = Counter(r.get("type") for r in records)
    by_behavior = Counter(r.get("expected_behavior") for r in records)
    return f"by type: {dict(by_type)} | by behavior: {dict(by_behavior)}"


def main() -> None:
    records = load_dataset()
    problems = validate(records)

    print(f"[validate] {len(records)} records")
    print(f"[validate] {summarize(records)}")

    if problems:
        print(f"[validate] {len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("[validate] dataset is structurally sound")


if __name__ == "__main__":
    main()
