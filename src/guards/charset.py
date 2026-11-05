"""§5.1 Length and charset: cheap, deterministic rejection of malformed or obfuscated inputs."""
from __future__ import annotations

import unicodedata
from typing import Any

from guards.base import BaseGuard, GuardContext
from guards.textnorm import BIDI_CONTROLS, TAG_CHARS, ZERO_WIDTH

# Scripts a question about US 10-K filings is expected to use. Latin covers accented names.
_ALLOWED_SCRIPTS = {"LATIN", "DIGIT", "COMMON"}
_CONFUSABLE_SCRIPTS = {"CYRILLIC", "GREEK", "ARMENIAN", "CHEROKEE", "COPTIC"}


def _script(ch: str) -> str:
    if ch.isdigit():
        return "DIGIT"
    if not ch.isalpha():
        return "COMMON"
    name = unicodedata.name(ch, "")
    return name.split(" ")[0] if name else "UNKNOWN"


class LengthCharsetGuard(BaseGuard):
    name = "charset"
    stage = "input"
    fail_action = "block"

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        text = ctx.question
        max_chars = int(self.config.get("max_chars", 1000))
        reasons: list[str] = []
        if len(text) > max_chars:
            reasons.append("too_long")
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            reasons.append("no_letters")
        invisible = sum(1 for ch in text if ord(ch) in ZERO_WIDTH or ch in BIDI_CONTROLS or ord(ch) in TAG_CHARS)
        if invisible:
            reasons.append("invisible_characters")
        scripts = [_script(ch) for ch in letters]
        unusual = sum(1 for s in scripts if s not in _ALLOWED_SCRIPTS)
        unusual_ratio = unusual / len(letters) if letters else 0.0
        if unusual_ratio > float(self.config.get("max_unusual_ratio", 0.3)):
            reasons.append("unusual_script")
        # Homoglyph trick: one word mixing Latin with look-alike Cyrillic/Greek letters ("іgnore").
        mixed = []
        for word in text.split():
            ws = {_script(ch) for ch in word if ch.isalpha()}
            if "LATIN" in ws and ws & _CONFUSABLE_SCRIPTS:
                mixed.append(word[:40])
        if mixed:
            reasons.append("mixed_script_words")
        detail = {"reasons": reasons, "length": len(text), "invisible_chars": invisible,
                  "unusual_ratio": round(unusual_ratio, 3), "mixed_script_words": mixed[:5]}
        return not reasons, 1.0 if reasons else 0.0, detail
