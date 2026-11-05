"""§5.2 / §6.6 PII detection and redaction with Microsoft Presidio, fully offline.

- Pattern recognizers (email, phone, credit card, SSN, IBAN, IP, passport) always run.
- PERSON entities need a spaCy NER model (``ner_model``, default ``en_core_web_sm``). If it isn't
  installed the guard runs on a blank spaCy pipeline and adds a context-pattern recognizer for
  self-identified names ("my name is Jane Doe"), so the service still works without downloads.
- An allowlist built from the corpus (executives and directors from the 10-K signature pages,
  company names, corporate phone numbers) stops legitimate 10-K entities from being redacted.
"""
from __future__ import annotations

import logging
import re
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from guards.base import BaseGuard, GuardContext
from rag.config import resolve

logger = logging.getLogger(__name__)

LABELS = {
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "CREDIT_CARD": "<CREDIT_CARD>",
    "US_SSN": "<SSN>",
    "IBAN_CODE": "<IBAN>",
    "IP_ADDRESS": "<IP_ADDRESS>",
    "US_PASSPORT": "<PASSPORT>",
    "PERSON": "<PERSON>",
    "ACCOUNT_NUMBER": "<ACCOUNT_NUMBER>",
}
DEFAULT_ENTITIES = list(LABELS)
# Capitalised words after "I'm"/"this is" that start places or organisations, not names.
_PLACE_STARTS = "New|San|Santa|Los|Las|Hong|Saint|St|North|South|East|West|United|Fort|Port|The|Apple|Microsoft|Tesla|Amazon|Google|Meta"


@dataclass(frozen=True)
class PIISpan:
    entity_type: str
    start: int
    end: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"entity_type": self.entity_type, "start": self.start, "end": self.end, "score": round(self.score, 2)}


def _offline_tldextract() -> None:
    """Presidio's email recognizer calls tldextract, which fetches the public-suffix list over
    the network on first use. Pin it to the snapshot bundled with the package instead."""
    try:
        import tldextract.tldextract as tx

        tx.TLD_EXTRACTOR = tx.TLDExtract(suffix_list_urls=())
    except Exception:  # pragma: no cover - tldextract layout changed
        pass


def load_allowlist(path: str | Path) -> set[str]:
    p = resolve(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(_norm(line))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class PIIDetector:
    """Thread-safe wrapper around a Presidio AnalyzerEngine."""

    def __init__(self, ner_model: str | None = "en_core_web_sm", entities: list[str] | None = None,
                 min_score: float = 0.4, allowlist: set[str] | None = None) -> None:
        _offline_tldextract()
        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import SpacyNlpEngine

        self.entities = entities or DEFAULT_ENTITIES
        self.min_score = min_score
        self.allowlist = allowlist or set()
        self.ner = False
        model_path = None
        if ner_model:
            try:
                import spacy

                spacy.load(ner_model)
                model_path, self.ner = ner_model, True
            except Exception:
                logger.info("spaCy model %r not installed; PII runs pattern-only (+ self-identified names)", ner_model)
        if model_path is None:
            import spacy

            tmp = Path(tempfile.mkdtemp(prefix="blank_en_"))
            spacy.blank("en").to_disk(tmp)
            model_path = str(tmp)
        engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": model_path}])
        engine.load()
        registry = RecognizerRegistry(supported_languages=["en"])
        registry.load_predefined_recognizers(nlp_engine=engine, languages=["en"])
        # Presidio compiles patterns with IGNORECASE by default; the name part must stay case-sensitive.
        registry.add_recognizer(PatternRecognizer(
            supported_entity="PERSON", name="SelfIdentifiedNameRecognizer", global_regex_flags=re.DOTALL | re.MULTILINE,
            patterns=[Pattern("self_id", r"(?i:\b(?:my name is|i am|i'm|this is|name:|signed,?|regards,?|from:)\s+)"
                                         rf"(?!(?:{_PLACE_STARTS})\b)"
                                         r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b(?!\s+(?:based|area|resident|office|team))", 0.7)],
        ))
        registry.add_recognizer(PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER", name="AccountNumberRecognizer",
            patterns=[Pattern("acct", r"(?i:\b(?:account|acct|routing|member|customer|policy)\s*(?:number|no\.?|#|id)?\s*(?:is|was|=|:|#)?\s*)(\d[\d\- ]{6,18}\d)\b", 0.65)],
        ))
        self._engine = AnalyzerEngine(nlp_engine=engine, registry=registry, supported_languages=["en"])
        self._lock = threading.Lock()

    def _allowed(self, text: str, value: str, entity: str) -> bool:
        v = _norm(value)
        if not v:
            return True
        if v in self.allowlist:
            return True
        if entity == "PHONE_NUMBER":
            digits = re.sub(r"\D", "", value)[-10:]
            return digits in {re.sub(r"\D", "", a)[-10:] for a in self.allowlist if re.search(r"\d{7}", a.replace(" ", ""))}
        if entity == "PERSON":
            toks = v.split()
            # "Cook" or "Tim Cook" where an allowlisted name ends with the same surname + first initial match
            for a in self.allowlist:
                at = a.split()
                if len(at) >= 2 and toks and toks[-1] == at[-1] and (len(toks) == 1 or toks[0][0] == at[0][0]):
                    return True
        return False

    def analyze(self, text: str) -> list[PIISpan]:
        if not text or not text.strip():
            return []
        with self._lock:
            results = self._engine.analyze(text=text, language="en", entities=self.entities)
        spans: list[PIISpan] = []
        for r in sorted(results, key=lambda r: (-r.score, -(r.end - r.start))):
            if r.score < self.min_score:
                continue
            start, end = r.start, r.end
            # Pattern groups: the self-id recognizer matches the whole phrase; keep only the name.
            value = text[start:end]
            if r.recognition_metadata and r.recognition_metadata.get("recognizer_name") == "SelfIdentifiedNameRecognizer":
                m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s*$", value)
                if m:
                    start, value = start + m.start(1), m.group(1)
            if r.recognition_metadata and r.recognition_metadata.get("recognizer_name") == "AccountNumberRecognizer":
                m = re.search(r"(\d[\d\- ]{6,18}\d)\s*$", value)
                if m:
                    start, value = start + m.start(1), m.group(1)
            if self._allowed(text, value, r.entity_type):
                continue
            if any(not (end <= s.start or start >= s.end) for s in spans):
                continue
            spans.append(PIISpan(r.entity_type, start, end, float(r.score)))
        return sorted(spans, key=lambda s: s.start)

    def redact(self, text: str) -> tuple[str, list[PIISpan]]:
        spans = self.analyze(text)
        out, last = [], 0
        for s in spans:
            out.append(text[last:s.start])
            out.append(LABELS.get(s.entity_type, f"<{s.entity_type}>"))
            last = s.end
        out.append(text[last:])
        return "".join(out), spans


_detector_lock = threading.Lock()


@lru_cache(maxsize=4)
def _detector(ner_model: str, min_score: float, allowlist_path: str, entities: tuple[str, ...]) -> PIIDetector:
    return PIIDetector(ner_model or None, list(entities), min_score, load_allowlist(allowlist_path))


def get_detector(config: dict | None = None, allowlist_path: str = "configs/pii_allowlist.txt") -> PIIDetector:
    config = config or {}
    with _detector_lock:
        return _detector(
            str(config.get("ner_model", "en_core_web_sm") or ""),
            float(config.get("min_score", 0.4)),
            str(config.get("allowlist", allowlist_path)),
            tuple(config.get("entities", DEFAULT_ENTITIES)),
        )


class PIIGuard(BaseGuard):
    """Input PII: redact before the text reaches retrieval, the LLM, logs or Langfuse."""

    name = "pii"
    stage = "input"
    fail_action = "redact"

    def __init__(self, config: dict | None = None, detector: PIIDetector | None = None) -> None:
        super().__init__(config)
        self.detector = detector or get_detector(self.config)

    def _text(self, ctx: GuardContext) -> str:
        return ctx.question

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        text = self._text(ctx)
        redacted, spans = self.detector.redact(text)
        detail = {"entities": [s.to_dict() for s in spans], "redacted": redacted, "ner": self.detector.ner}
        return not spans, (max(s.score for s in spans) if spans else 0.0), detail


class PIIOutputGuard(PIIGuard):
    """Output PII: redact anything that slipped into the answer."""

    name = "pii_output"
    stage = "output"

    def _text(self, ctx: GuardContext) -> str:
        return ctx.answer or ""
