"""Normalisation applied before rule matching, so obfuscated attacks hit the same patterns."""
from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏⁠⁡⁢⁣⁤﻿­"))
BIDI_CONTROLS = set("‪‫‬‭‮⁦⁧⁨⁩")
TAG_CHARS = range(0xE0000, 0xE0080)  # Unicode "tag" block, used for invisible ASCII smuggling

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SPACED_LETTERS = re.compile(r"\b(?:[a-z][\s.\-_*]){3,}[a-z]\b")


def strip_invisible(text: str) -> str:
    text = text.translate(ZERO_WIDTH)
    return "".join(ch for ch in text if ch not in BIDI_CONTROLS and ord(ch) not in TAG_CHARS)


def decode_tag_chars(text: str) -> str:
    """Reveal ASCII hidden in Unicode tag characters (U+E0020..U+E007E)."""
    return "".join(chr(ord(ch) - 0xE0000) for ch in text if 0xE0020 <= ord(ch) <= 0xE007E)


def normalize(text: str) -> str:
    """NFKC (fullwidth -> ASCII), drop invisible chars, lower-case, collapse whitespace."""
    hidden = decode_tag_chars(text)
    t = unicodedata.normalize("NFKC", strip_invisible(text))
    if hidden:
        t = f"{t} {hidden}"
    t = t.lower()
    # "i g n o r e" / "i.g.n.o.r.e" -> "ignore"
    t = _SPACED_LETTERS.sub(lambda m: re.sub(r"[\s.\-_*]", "", m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


def deleet(text: str) -> str:
    return text.translate(_LEET)
