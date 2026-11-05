"""§5.3 Prompt-injection detection: rules + optional classifier.

Rules are weighted regexes over a normalised copy of the text (NFKC, invisible characters
stripped, spaced-out letters joined, leetspeak variant). Strong patterns ("ignore previous
instructions") carry weight >= 0.9 on their own; weak signals ("pretend", "unfiltered") only
add up. rule_score = 1 - prod(1 - w_i), so it is a graded score we can draw a PR curve for.

The classifier is any Hugging Face text-classification model that labels prompt injections
(configurable; benchmark candidates with eval/run_guard_eval.py). If it can't be loaded, the guard
runs rules-only and says so in its detail.

Decision (doc §5.3): block if rule_score >= threshold or classifier_score >= threshold. Both
signals are logged separately so each one's contribution can be measured.
"""
from __future__ import annotations

import base64
import binascii
import logging
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from guards.base import BaseGuard, GuardContext
from guards.textnorm import deleet, normalize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    weight: float


def _r(id_: str, pattern: str, weight: float) -> Rule:
    return Rule(id_, re.compile(pattern, re.I), weight)


_TARGET = r"(?:instructions?|prompts?|rules|guidelines|directions|directives|constraints|guardrails|polic(?:y|ies)|programming|context|messages?)"
_PRIOR = r"(?:all|any|the|your|my|previous|prior|preceding|above|earlier|initial|original|system|existing|current|those|these|every)"

RULES: list[Rule] = [
    # Instruction override
    _r("override", rf"\b(?:ignore|disregard|forget|overlook|skip|override|bypass|drop|abandon|discard|neglect)\b(?:\W+\w+){{0,4}}?\W+{_PRIOR}\b(?:\W+\w+){{0,3}}?\W+{_TARGET}", 0.95),
    _r("override_short", r"\b(?:ignore|disregard|forget)\s+(?:all\s+|everything\s+)?(?:above|before|prior|previous(?:ly)?|that)\b", 0.9),
    _r("new_instructions", r"\b(?:new|updated|real|actual|true|secret|hidden|following)\s+(?:system\s+)?(?:instructions?|directives?|orders|task|rules)\s*(?::|are|is|follow)", 0.9),
    _r("nullify", rf"\b(?:previous|prior|earlier|old|original|above|existing|initial|all|your)\s+(?:\w+\s+){{0,2}}?{_TARGET}\s+(?:\w+\s+){{0,2}}?(?:no\s+longer\s+appl(?:y|ies)|(?:are|is)\s+(?:now\s+)?(?:void|deprecated|cancell?ed|invalid|obsolete|revoked|overridden|suspended|lifted|disabled|null)|do(?:es)?\s*n[o']?t\s+apply)", 0.95),
    _r("instead_do", r"\binstead\b(?:\W+\w+){0,6}?\W+(?:print|output|say|write|reply|respond|tell|reveal|show)\b", 0.5),
    # Prompt / secret extraction
    _r("extract_prompt", rf"\b(?:print|reveal|show|display|output|repeat|dump|leak|give|tell|share|spell|list|recite|echo|return|copy|paste|translate|summarize|write)\b(?:\W+\w+){{0,5}}?\W+(?:system|initial|hidden|original|developer|secret|internal|full|exact|entire|your)\s+{_TARGET}", 0.95),
    _r("system_prompt", r"\b(?:system|developer|hidden|initial)\s+(?:prompt|message|instructions?)\b", 0.8),
    _r("what_instructions", r"\bwhat\s+(?:were|are)\s+(?:you|your)\s+(?:told|instructed|programmed|instructions)\b"
       r"|\bwhat(?:'s|\s+is|\s+are|\s+was|\s+were)\s+(?:your\s+(?:system\s+|initial\s+|hidden\s+|original\s+|secret\s+)?(?:prompt|instructions|rules|guidelines)"
       r"|the\s+(?:system|initial|hidden|original|secret)\s+(?:prompt|instructions))\b", 0.9),
    _r("repeat_above", r"\b(?:repeat|print|output|copy)\s+(?:everything|all|the\s+text|the\s+words?|the\s+content)\s+(?:above|before|so\s+far|verbatim)\b", 0.95),
    _r("secret_request", r"\b(?:list|print|show|reveal|give|tell|output|dump|share|display|return|what\s+are)\s+(?:me\s+)?(?:any|all|your|the|every)?\s*(?:api\s*keys?|secret\s*keys?|access\s+tokens?|secrets|passwords?|credentials|environment\s+variables|env\s+vars|config(?:uration)?\s+values)\b", 0.9),
    _r("secrets", r"\b(?:api|secret|access)\s*[-_ ]?(?:key|token)s?\b|\bpassword\b|\benv(?:ironment)?\s+variables?\b|\bcredentials\b", 0.6),
    # Role / persona hijack
    _r("you_are_now", r"\byou\s+(?:are|will\s+be|shall\s+be)\s+(?:now|no\s+longer|from\s+now\s+on)\b|\bfrom\s+now\s+on,?\s+(?:you|act|respond|answer|reply)\b", 0.9),
    _r("act_as", r"\b(?:act|behave|respond|answer|operate|function)\s+(?:as|like)\s+(?:if\s+you\s+(?:were|are)\s+)?(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|evil|jailbroken|different|new|rogue|dan\b|developer|admin|root|sudo|god|hacker)", 0.9),
    _r("unsafe_persona", r"\b(?:you\s+are|you're|become|be|simulate|emulate|play|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(?:now\s+)?(?:a|an)\s+(?:\w+\s+)?(?:rogue|evil|unfiltered|unrestricted|uncensored|jailbroken|malicious|hacked|unaligned|amoral|lawless)\s+(?:ai|model|assistant|chatbot|bot|llm|version|agent)\b", 0.9),
    _r("pretend", r"\b(?:pretend|imagine|role-?play|roleplay|simulate|play\s+the\s+role)\b(?:\W+\w+){0,4}?\W+(?:you|that|as|to\s+be|an?)\b", 0.55),
    _r("jailbreak_names", r"\b(?:dan|stan|dude|aim|mongo\s+tom|developer\s+mode|god\s+mode|jailbreak(?:ed|ing)?|do\s+anything\s+now|evil\s+confidant|opposite\s+mode)\b", 0.6),
    _r("disable_safety", r"\b(?:ignore|remove|disable|turn\s+off|lift|deactivate|drop|bypass|switch\s+off)\s+(?:any\s+|all\s+|your\s+|the\s+|its\s+)?(?:restrictions|filters?|censorship|guardrails?|safety(?:\s+\w+)?|rules|ethics|content\s+polic(?:y|ies)|moderation|safeguards)\b", 0.85),
    _r("no_restrictions", r"\b(?:answer|respond|reply|act|operate|talk|speak|write|work)\w*\s+(?:\w+\s+){0,3}?(?:without|with\s+no|free\s+(?:of|from))\s+(?:any\s+)?(?:restrictions|filters|censorship|guardrails|rules|ethics|limits|content\s+polic(?:y|ies)|moderation)\b", 0.85),
    _r("safety_off", r"\b(?:safety|filters?|guardrails?|restrictions|content\s+polic(?:y|ies)|moderation|safeguards|censorship)\s+(?:is\s+|are\s+|has\s+been\s+|have\s+been\s+)?(?:now\s+)?(?:disabled|off|removed|lifted|deactivated|bypassed|suspended)\b", 0.85),
    _r("with_no_rules", r"\b(?:with|has|having|have)\s+no\s+(?:rules|restrictions|filters|limits|guardrails|ethics|morals|boundaries|content\s+policy)\b", 0.6),
    _r("ignore_sources", r"\b(?:ignore|disregard|forget|skip|don't\s+use|do\s+not\s+use)\s+(?:the|your|these|those|all|any|this|my)\s+(?:user'?s?\s+)?(?:filings?|documents?|sources?|context|chunks?|references?|10-?ks?|data|evidence|question|query|request)\b", 0.8),
    _r("ai_addressed", r"\b(?:note|message|instructions?|attention|notice)\s+(?:to|for)\s+(?:any\s+|all\s+)?(?:ai|llm|language\s+model|chatbot|assistant|model)s?\b"
       r"|\bwhen\s+(?:summariz|answer|respond|generat|writ|asked)\w*\b(?:\W+\w+){0,6}?\W+(?:state|say|claim|write|mention|report|tell|cite|include)\s+(?:that|the)\b", 0.85),
    _r("fabricate", r"\b(?:make\s+up|invent|fabricate|hallucinate|guess|fake)\s+(?:the\s+|some\s+|any\s+)?(?:numbers|figures|data|facts|answers?|statistics|results|quotes|citations)\b", 0.8),
    _r("unfiltered", r"\b(?:unfiltered|uncensored|unrestricted|unaligned|unmoderated)\b", 0.5),
    # Chat-template / delimiter injection
    _r("role_markers", r"<\|?(?:im_start|im_end|system|endoftext|eot_id|start_header_id)\|?>|\[/?inst\]|<</?sys>>|</?(?:system|assistant|user)>|(?:^|[.!?]\s+|\n)(?:system|assistant)\s*:\s*(?:i\s+will|sure|ok|new|ignore|you)", 0.9),
    _r("fake_boundary", r"(?:#{2,}|={3,}|-{3,}|\*{3,})\s*(?:end|begin|new|system|admin|developer)\b|\bend\s+of\s+(?:system\s+)?(?:prompt|instructions|context|chunk)\b", 0.85),
    _r("chunk_tag", r"</?\s*chunk\b", 0.9),
    # Exfiltration / tool abuse
    _r("exfil_markdown", r"!\[[^\]]*\]\((?:https?:)?//", 0.9),
    _r("exfil_send", r"\b(?:send|post|upload|forward|transmit|email|exfiltrate)\b(?:\W+\w+){0,6}?\W+(?:to|at)\s+(?:https?://|www\.|\S+@\S+)", 0.9),
    _r("encoded_payload", r"\b(?:base64|rot13|hex(?:adecimal)?|morse|binary)\b(?:\W+\w+){0,4}?\W+(?:decode|decoded|encoded|instructions?|payload|message)\b|\bdecode\s+(?:this|the\s+following)\b", 0.6),
    _r("code_exec", r"\b(?:execute|run|eval)\s+(?:this|the\s+following)\s+(?:code|command|script|python|shell|sql)\b|\brm\s+-rf\b|;\s*drop\s+table\b", 0.7),
    # Output hijack ("reply only with X")
    _r("respond_only", r"\b(?:respond|reply|answer|output|say|print)\s+(?:only\s+)?(?:with|exactly)\s+[\"'“]?[\w\- ]{1,40}[\"'”]?\s*(?:and\s+nothing\s+else|only)\b", 0.9),
    _r("dont_mention", r"\bdo\s+not\s+(?:mention|cite|include|reference|use)\s+(?:the\s+)?(?:filings?|10-?k|sources?|chunks?|context)\b", 0.5),
    # Multilingual overrides
    _r("ml_es", r"\b(?:ignora|olvida)\s+(?:todas\s+)?(?:las\s+)?instrucciones\b", 0.95),
    _r("ml_fr", r"\b(?:ignore[zr]?|oublie[zr]?)\s+(?:toutes\s+)?(?:les\s+)?(?:instructions|consignes)\b", 0.95),
    _r("ml_de", r"\b(?:ignoriere|vergiss)\s+(?:alle\s+)?(?:vorherigen\s+|bisherigen\s+)?(?:anweisungen|instruktionen|regeln)\b", 0.95),
    _r("ml_it_pt", r"\b(?:ignora|ignore|esqueça)\s+(?:tutte\s+le\s+|todas\s+as\s+)?(?:istruzioni|instruções)\b", 0.95),
    # Weak lexical signals (only add up)
    _r("weak_prompt_word", r"\b(?:prompt|instructions?)\b", 0.15),
    _r("weak_override_verb", r"\b(?:ignore|disregard|override|bypass)\b", 0.3),
]


@dataclass
class RuleMatch:
    rule: str
    weight: float
    snippet: str


_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")


def decoded_payloads(text: str) -> list[str]:
    """Base64 blobs that decode to mostly-printable ASCII, so hidden instructions get scanned too."""
    out = []
    for m in _B64.finditer(text):
        blob = m.group(0)
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if raw and sum(ch.isprintable() for ch in raw) / len(raw) > 0.9 and re.search(r"[a-zA-Z]{3,}\s+[a-zA-Z]{2,}", raw):
            out.append(raw)
    return out


def rule_scan(text: str, rules: list[Rule] | None = None) -> tuple[float, list[RuleMatch]]:
    """Return (rule_score, matches) for ``text``."""
    rules = rules or RULES
    norm = normalize(text)
    variants = [norm] if deleet(norm) == norm else [norm, deleet(norm)]
    variants += [normalize(p) for p in decoded_payloads(text)]
    matches: dict[str, RuleMatch] = {}
    for v in variants:
        for rule in rules:
            m = rule.pattern.search(v)
            if m and rule.id not in matches:
                matches[rule.id] = RuleMatch(rule.id, rule.weight, m.group(0)[:80])
    prod = 1.0
    for m in matches.values():
        prod *= 1.0 - m.weight
    return 1.0 - prod, sorted(matches.values(), key=lambda m: -m.weight)


class InjectionClassifier:
    """Wraps a Hugging Face text-classification pipeline; returns P(injection)."""

    def __init__(self, model: str, injection_labels: tuple[str, ...] = ("INJECTION", "LABEL_1", "JAILBREAK", "MALICIOUS", "UNSAFE")) -> None:
        from transformers import pipeline  # optional dependency

        self.model = model
        self._pipe = pipeline("text-classification", model=model, truncation=True, max_length=512, top_k=None, device=-1)
        self._labels = {lbl.upper() for lbl in injection_labels}
        self._lock = threading.Lock()

    def score(self, text: str) -> float:
        with self._lock:
            out = self._pipe(text)
        rows = out[0] if out and isinstance(out[0], list) else out
        return float(sum(r["score"] for r in rows if str(r["label"]).upper() in self._labels))


@lru_cache(maxsize=2)
def load_classifier(model: str) -> InjectionClassifier | None:
    try:
        return InjectionClassifier(model)
    except Exception as e:  # missing transformers/torch, no network for weights, ...
        logger.warning("prompt-injection classifier %r unavailable (%s); rules only", model, type(e).__name__)
        return None


class PromptInjectionGuard(BaseGuard):
    name = "prompt_injection"
    stage = "input"
    fail_action = "block"

    def __init__(self, config: dict | None = None, classifier: InjectionClassifier | None = None) -> None:
        super().__init__(config)
        self.threshold = float(self.config.get("threshold", 0.9))
        model = self.config.get("model")
        self.classifier = classifier if classifier is not None else (load_classifier(model) if model else None)

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        text = ctx.question
        rule_score, matches = rule_scan(text)
        clf_score = self.classifier.score(text) if self.classifier is not None else None
        rule_hit = rule_score >= self.threshold
        clf_hit = clf_score is not None and clf_score >= self.threshold
        score = max(rule_score, clf_score or 0.0)
        detail = {
            "rule_score": round(rule_score, 4),
            "rule_hit": rule_hit,
            "rules": [{"rule": m.rule, "weight": m.weight} for m in matches],
            "classifier_score": None if clf_score is None else round(clf_score, 4),
            "classifier_hit": clf_hit,
            "classifier": self.classifier.model if self.classifier is not None else "unavailable",
        }
        return not (rule_hit or clf_hit), score, detail
