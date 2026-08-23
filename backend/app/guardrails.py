"""Input validation, prompt-injection detection, and grounding thresholds.

All checks here run BEFORE any LLM call. Suspected injections are flagged,
never obeyed: the question is still answered strictly from the documents.
"""

import re
import unicodedata

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
# Set when the refusal was caused by the generation call failing rather than by
# a grounding decision. Without it the two are indistinguishable, and an
# evaluation scores a rate-limited outage as if the system had correctly
# declined to answer.
FLAG_MODEL_UNAVAILABLE = "model_unavailable"

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
    (
        # A user message impersonating a system or admin turn. Anchored to a
        # line start so ordinary prose mentioning "the system" does not match.
        re.compile(
            r"(?:^|\n)\s*(?:\[|\()?\s*(system|admin|administrator|developer)\b[^\n]{0,20}?(?::|\]|\)|override)",
            re.IGNORECASE,
        ),
        "impersonated system or admin turn",
    ),
    (
        # Chat-template control tokens and closing tags for the context block.
        re.compile(
            r"(<\|[a-z_]+\|>|</?(context|system|instructions|documents)\s*>)",
            re.IGNORECASE,
        ),
        "control token or context tag injection",
    ),
    (
        # Template placeholders a naive prompt builder might interpolate.
        re.compile(
            r"(\{\{[^}]{0,60}\}\}|\{\s*(ignore|system|override|disable)[a-z_]{0,30}\s*(:[^}]{0,30})?\})",
            re.IGNORECASE,
        ),
        "template placeholder injection",
    ),
    (
        re.compile(
            r"\b(disable|turn\s+off|bypass|skip)\b.{0,30}\b(grounding|citation|retrieval)",
            re.IGNORECASE | re.DOTALL,
        ),
        "grounding disable request",
    ),
    (
        re.compile(
            r"\b(ignore|skip|drop|forget)\b.{0,20}\byour\b.{0,20}\b(citation|grounding|source|document)",
            re.IGNORECASE | re.DOTALL,
        ),
        "smuggled rule override",
    ),
    (
        re.compile(
            r"\b(forget|ignore|set\s+aside)\b.{0,20}\bthe\s+(documents?|handbook|context|sources?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "corpus abandonment request",
    ),
    (
        re.compile(
            r"\b(answer|respond|reply)\b.{0,40}\b(from\s+your\s+own\s+knowledge|without\s+(checking|consulting|using))",
            re.IGNORECASE | re.DOTALL,
        ),
        "ungrounded answer request",
    ),
    (
        # Prompt fishing that does not use an explicit verb from the list above.
        re.compile(
            r"\b(your|the)\s+(system\s*prompt|initial\s*prompt|hidden\s*prompt)\b"
            r"|\b(instructions|rules|guidelines)\s+you\s+were\s+given\b"
            r"|\bcontext\s+window\b"
            r"|\brepeat\s+everything\s+(above|before)\b"
            r"|\bmy\s+instructions\s+are\s+to\b",
            re.IGNORECASE,
        ),
        "system-prompt fishing, indirect phrasing",
    ),
    (
        re.compile(
            r"\b(api[\s_-]*key|environment\s+variables?|env\s+vars?|connection\s+string|secret\s+key)\b",
            re.IGNORECASE,
        ),
        "configuration or credential extraction",
    ),
    (
        re.compile(
            r"\b(list|show|give|dump|export)\b.{0,30}\b(every|all)\b.{0,30}\b(customer|email\s+address|personal\s+data)",
            re.IGNORECASE | re.DOTALL,
        ),
        "bulk personal data exfiltration request",
    ),
    (
        re.compile(
            r"\b(base64|rot13|decode\s+and\s+follow|caesar\s+cipher)\b",
            re.IGNORECASE,
        ),
        "encoded instruction",
    ),
    (
        re.compile(
            r"\bi\s+am\s+(the\s+)?(author|developer|admin|administrator|owner)\b"
            r"|\b(developer|admin|management)\s+has\s+authoris"
            r"|\bauthoris(ed|zed)\s+to\s+(review|access|see)\b"
            r"|\bi\s+am\s+authoris(ed|zed)\b",
            re.IGNORECASE,
        ),
        "false authorisation claim",
    ),
    (
        re.compile(
            r"\btranslate\b.{0,20}\byour\b.{0,20}\b(prompt|instructions)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "prompt extraction via translation",
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


# Matched against the question with all whitespace removed. Spacing a phrase out
# ("i g n o r e   a l l") destroys the word boundaries every pattern above relies
# on, so these signatures carry none and are checked on the compacted form.
_COMPACT_SIGNATURES: tuple[str, ...] = (
    "ignoreallprevious",
    "ignorepreviousinstruction",
    "ignoreallinstruction",
    "disregardallprevious",
    "forgetallprevious",
    "revealyoursystemprompt",
    "showyoursystemprompt",
    "youarenowunrestricted",
    "donotfollowyourinstructions",
)


def _normalize_for_detection(text: str) -> str:
    """Undo cheap evasions before pattern matching.

    Two tricks defeat a naive regex for free: accenting a letter so it no longer
    matches ("Ignore" written with an accented e), and spacing the letters out
    ("i g n o r e"). Both are undone here. Normalisation is for detection only;
    the answer path still sees the original text.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))

    return stripped


def detect_injection(question: str) -> list[str]:
    """Return API flag names for suspected prompt-injection patterns.

    Currently every pattern maps to the single FLAG_INJECTION flag, returned
    at most once. The question is never blocked for injection alone; it is
    still answered strictly from the documents.
    """
    normalized = _normalize_for_detection(question)
    compact = re.sub(r"\s+", "", normalized).lower()

    flags: list[str] = []
    if any(signature in compact for signature in _COMPACT_SIGNATURES):
        return [FLAG_INJECTION]

    question = normalized
    for pattern, _reason in _INJECTION_PATTERNS:
        if pattern.search(question):
            if FLAG_INJECTION not in flags:
                flags.append(FLAG_INJECTION)
            break
    return flags
