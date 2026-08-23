"""Structural validation of the evaluation dataset.

Runs before any metric does. A label that points at a chunk id which no longer
exists is worse than a missing label, because it silently scores against
nothing and the eval still reports a number. This catches that, and it has to
keep passing every time the corpus is re-chunked.

Standalone:  python -m tests.eval.validate_dataset
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

REQUIRED_FIELDS = {
    "id",
    "question",
    "ground_truth",
    "source_chunks",
    "expected_behavior",
    "type",
}
VALID_BEHAVIORS = {"answer", "refuse"}
VALID_TYPES = {"factual", "multi_hop", "unanswerable", "ambiguous"}


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


def validate(records: list[dict], known_chunk_ids: set[str]) -> list[str]:
    """Return a list of problems. Empty means the dataset is structurally sound."""
    problems: list[str] = []

    id_counts = Counter(r.get("id") for r in records)
    for rid, count in id_counts.items():
        if count > 1:
            problems.append(f"duplicate id {rid} appears {count} times")

    seen_questions: dict[str, str] = {}

    for record in records:
        rid = record.get("id", "<missing id>")

        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            problems.append(f"{rid}: missing fields {sorted(missing)}")
            continue

        if record["expected_behavior"] not in VALID_BEHAVIORS:
            problems.append(
                f"{rid}: expected_behavior {record['expected_behavior']!r} "
                f"not in {sorted(VALID_BEHAVIORS)}"
            )

        if record["type"] not in VALID_TYPES:
            problems.append(f"{rid}: type {record['type']!r} not in {sorted(VALID_TYPES)}")

        question = record["question"].strip().lower()
        if question in seen_questions:
            problems.append(f"{rid}: question duplicates {seen_questions[question]}")
        else:
            seen_questions[question] = rid

        chunks = record["source_chunks"]
        if not isinstance(chunks, list):
            problems.append(f"{rid}: source_chunks must be a list")
            continue

        # An unanswerable question is defined by having no supporting chunk.
        # Anything else must cite at least one, or there is nothing to score
        # context recall against.
        if record["expected_behavior"] == "refuse":
            if chunks:
                problems.append(
                    f"{rid}: expected_behavior is refuse but cites {len(chunks)} chunk(s)"
                )
        elif not chunks:
            problems.append(f"{rid}: expected_behavior is answer but cites no chunks")

        for chunk_id in chunks:
            if chunk_id not in known_chunk_ids:
                problems.append(f"{rid}: source chunk {chunk_id!r} does not exist")

    return problems


def main() -> None:
    from app.ingest import build_chunks

    ids, _ = build_chunks()
    known = set(ids)
    records = load_dataset()

    problems = validate(records, known)

    by_type = Counter(r.get("type") for r in records)
    by_behavior = Counter(r.get("expected_behavior") for r in records)
    print(f"[validate] {len(records)} records across {len(known)} known chunks")
    print(f"[validate] by type: {dict(by_type)}")
    print(f"[validate] by behavior: {dict(by_behavior)}")

    if problems:
        print(f"[validate] {len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("[validate] dataset is structurally sound")


if __name__ == "__main__":
    main()
