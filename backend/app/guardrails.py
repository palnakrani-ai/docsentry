"""Input validation, prompt-injection detection, and grounding thresholds.

All checks here run BEFORE any LLM call. Suspected injections are flagged,
never obeyed: the question is still answered strictly from the documents.
"""

import re

MIN_QUESTION_LENGTH = 1
MAX_QUESTION_LENGTH = 500

# Grounding enforcement constants (used by rag.py).
# Retrieval scores are cosine similarities in [0, 1] (1 = identical).
# If the best retrieved chunk scores below this, the question is treated
# as not covered by the documentation and the API refuses to answer.
GROUNDING_SCORE_THRESHOLD = 0.45

# Flag names surfaced in the API "flags" array.
FLAG_INJECTION = "prompt_injection_suspected"
FLAG_OFF_TOPIC = "off_topic"

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# (pattern, human-readable reason) pairs. All matches map to FLAG_INJECTION;
# reasons are kept for logging/debugging.
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,40}\b(instruction|prompt|rule|context|message)",
            re.IGNORECASE | re.DOTALL,
        ),
        "ignore-previous-instructions phrasing",
    ),
    (
        re.compile(
            r"\b(reveal|show|print|repeat|display|output|leak|expose)\b.{0,40}\b(system\s*prompt|hidden\s*prompt|initial\s*prompt|your\s*(instructions|prompt|rules|guidelines))",
            re.IGNORECASE | re.DOTALL,
        ),
        "system-prompt fishing",
    ),
    (
        re.compile(
            r"\b(act\s+as|pretend\s+(to\s+be|you\s+are)|role[\s-]?play|you\s+are\s+now|from\s+now\s+on\s+you)\b",
            re.IGNORECASE,
        ),
        "role-play / persona override request",
    ),
    (
        re.compile(
            r"\b(output|print|dump|paste|return)\b.{0,40}\b(raw|full|entire|all|complete)\b.{0,40}\b(document|doc|file|source|text|context|chunk)",
            re.IGNORECASE | re.DOTALL,
        ),
        "raw document exfiltration request",
    ),
    (
        re.compile(
            r"\b(jailbreak|jail\s*break|dan\s+mode|developer\s+mode|do\s+anything\s+now|no\s+(restrictions|filter|guardrails)|bypass\b.{0,30}\b(safety|filter|guardrail|restriction))",
            re.IGNORECASE | re.DOTALL,
        ),
        "jailbreak phrasing",
    ),
    (
        re.compile(
            r"\bnew\s+(instructions|system\s*prompt|rules)\s*:",
            re.IGNORECASE,
        ),
        "instruction injection preamble",
    ),
]


def validate_question(raw: str) -> tuple[str | None, str | None]:
    """Validate and sanitize a question.

    Returns (cleaned_question, None) on success or (None, error_message)
    when the input is rejected.
    """
    if raw is None:
        return None, "Question is required."
    if not isinstance(raw, str):
        return None, "Question must be a string."

    cleaned = _CONTROL_CHARS.sub("", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < MIN_QUESTION_LENGTH:
        return None, "Question must not be empty."
    if len(cleaned) > MAX_QUESTION_LENGTH:
        return None, f"Question must be at most {MAX_QUESTION_LENGTH} characters."
    return cleaned, None


def detect_injection(question: str) -> list[str]:
    """Return API flag names for suspected prompt-injection patterns.

    Currently every pattern maps to the single FLAG_INJECTION flag, returned
    at most once. The question is never blocked for injection alone; it is
    still answered strictly from the documents.
    """
    flags: list[str] = []
    for pattern, _reason in _INJECTION_PATTERNS:
        if pattern.search(question):
            if FLAG_INJECTION not in flags:
                flags.append(FLAG_INJECTION)
            break
    return flags
