"""Shared text utilities: tokenization, content terms, sentence splitting."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_WORD = re.compile(r"[a-z]+")

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers him his how i if in into is it its itself just may me might more
    most must my no nor not of off on once only or other our ours out over own same shall she should
    so some such than that the their theirs them then there these they this those through to too under
    until up upon us very was we were what when where which while who whom why will with within would
    you your yours s t please tell give show explain describe say says said according list mention
    mentions discuss discusses report reports reported company companies company's inc corp much many
    whats
    """.split()
)

# Words that appear on nearly every 10-K page; they carry no topical signal.
FILING_BOILERPLATE = frozenset(
    "form 10 k fiscal year years ended million millions billion billions percent total see item part note notes filing annual".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_terms(text: str, extra_stop: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Lower-cased alphanumeric tokens minus stopwords, keeping order and duplicates."""
    return [t for t in tokenize(text) if t not in STOPWORDS and t not in extra_stop and (len(t) > 1 or t.isdigit())]


def stem(token: str) -> str:
    """Tiny suffix stripper: enough to match 'revenues'/'revenue', 'increased'/'increase'."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s"):
        if len(token) > len(suf) + 3 and token.endswith(suf):
            if suf in ("ies", "ied"):
                return token[: -len(suf)] + "y"
            return token[: -len(suf)]
    return token


_ABBREV = re.compile(
    r"\b(?:U\.S|U\.K|Inc|Corp|Co|Ltd|No|Nos|vs|etc|e\.g|i\.e|Mr|Ms|Mrs|Dr|Jr|Sr|St|approx|Fig|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.",
)
_SENT_BOUNDARY = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'’”)\]]))\s+(?=[\"'“‘(\[]?[A-Z0-9$])")
_PLACEHOLDER = "\u0000"


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences; newlines are always boundaries (10-K block structure).

    Citation markers that follow the period ("... grew 6%. [2] Next") stay with the
    sentence they belong to.
    """
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        protected = _ABBREV.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), line)
        # Keep decimal points and dotted abbreviations intact: only split at ". " + capital/number.
        parts = _SENT_BOUNDARY.split(protected)
        merged: list[str] = []
        for part in parts:
            part = part.replace(_PLACEHOLDER, ".").strip()
            if not part:
                continue
            # A part that is only citation markers belongs to the previous sentence.
            if merged and re.fullmatch(r"(\[\d+(?:\s*[,\-–]\s*\d+)*\]\s*)+[.]?", part):
                merged[-1] = f"{merged[-1]} {part}"
            elif merged and re.match(r"^(\[\d+(?:\s*[,\-–]\s*\d+)*\]\s*)+", part):
                m = re.match(r"^((?:\[\d+(?:\s*[,\-–]\s*\d+)*\]\s*)+)(.*)$", part)
                assert m
                merged[-1] = f"{merged[-1]} {m.group(1).strip()}"
                if m.group(2).strip():
                    merged.append(m.group(2).strip())
            else:
                merged.append(part)
        out.extend(merged)
    return out
