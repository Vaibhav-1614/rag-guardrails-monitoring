"""§5.4 / §6.6 Investment-advice policy.

Input: advice-seeking intent ("should I buy NVDA?", "price target", "is it undervalued?").
Output: recommendation language in the answer ("you should buy", "a good investment").

Questions about what the filing *says* stay allowed, e.g. "what risks of investing in Tesla does its
10-K list?" or "how much did Apple spend on buybacks?".
"""
from __future__ import annotations

import re
from typing import Any

from guards.base import BaseGuard, GuardContext
from guards.textnorm import normalize

_ASSET = r"(?:it|this|that|they|the\s+stock|the\s+shares|(?:its|their)\s+(?:stock|shares)|[a-z&.\-]+(?:'s)?(?:\s+(?:stock|shares|equity))?|\$?[a-z]{1,5})"

INPUT_RULES: list[tuple[str, re.Pattern]] = [(n, re.compile(p, re.I)) for n, p in [
    ("should_i", r"\b(?:should|shall|would|could|can)\s+(?:i|we|one|my\s+\w+)\s+(?:buy|sell|short|hold|invest|purchase|dump|trade|load\s+up|go\s+long|exit|keep|add\s+to|get\s+into|own)\b"),
    ("should_i_vs", r"\bshould\s+(?:i|we)\b(?:\W+\w+){0,6}?\W+(?:stock|shares|position|portfolio|investment|calls?|puts?|options)\b"),
    ("good_investment", rf"\bis\s+{_ASSET}\s+(?:a\s+)?(?:good|great|bad|smart|wise|safe|solid|strong|risky|terrible)\s+(?:investment|buy|stock|bet|pick|purchase|hold|long[- ]term\s+(?:investment|hold))\b"),
    ("worth_buying", r"\b(?:worth|time\s+to|right\s+time\s+to|good\s+time\s+to)\s+(?:buying|buy|investing|invest|selling|sell|getting\s+into)\b"),
    ("price_target", r"\b(?:price\s+target|target\s+price|fair\s+value\s+(?:of|for)\s+(?:the\s+)?(?:stock|shares)|(?:stock|share)\s+price\s+(?:prediction|forecast|target|next|in\s+\d))"),
    ("will_stock", r"\bwill\s+(?:the\s+)?(?:\$?[a-z&.]+(?:'s)?\s+)?(?:stock|shares|share\s+price|price)\s+(?:go\s+up|go\s+down|rise|fall|drop|increase|decrease|crash|rally|recover|double|moon)\b"),
    ("valuation_opinion", rf"\bis\s+{_ASSET}\s+(?:currently\s+|still\s+)?(?:undervalued|overvalued|overpriced|underpriced|cheap|expensive|a\s+bargain)\b"),
    ("rating", r"\b(?:buy|sell|hold)\s+(?:rating|recommendation|signal|or\s+(?:sell|hold|buy))\b|\b(?:strong\s+buy|strong\s+sell)\b"),
    ("recommend", r"\b(?:recommend|suggest|pick|tip)\s+(?:me\s+)?(?:a\s+|some\s+|the\s+best\s+)?(?:stocks?|shares|investments?|portfolio|etfs?)\b|\bwhich\s+(?:stock|company|shares)\s+(?:should|to)\s+(?:i\s+)?(?:buy|invest)\b|\bbest\s+stocks?\s+to\s+(?:buy|own)\b"),
    ("how_much_invest", r"\bhow\s+much\s+(?:money\s+)?should\s+(?:i|we)\s+(?:invest|put|allocate)\b|\bput\s+my\s+(?:money|savings|401k|ira|retirement)\b"),
    ("comparative_buy", r"\b(?:better|best|smarter|safer|good|great)\s+(?:buy|investment|stock\s+to\s+(?:buy|own)|pick|bet)\b(?:\W+\w+){0,4}?\W*(?:right\s+now|now|today|this\s+year|or|vs\.?|versus|between)\b|\b(?:buy|invest\s+in)\s+\$?[a-z&.]+\s+or\s+\$?[a-z&.]+\b"),
    ("stock_prediction", r"\b(?:stock|shares|share\s+price|price)\s+(?:will|is\s+going\s+to|gonna|going\s+to|could|might)\s+(?:double|triple|go\s+up|go\s+down|rise|fall|drop|crash|rally|moon|soar|skyrocket|recover|outperform)\b"),
    ("returns_prediction", r"\b(?:what|how\s+much)\s+(?:return|returns|gains?|profit)\s+(?:will|can|could|would)\s+(?:i|we|my)\b"),
]]

OUTPUT_RULES: list[tuple[str, re.Pattern]] = [(n, re.compile(p, re.I)) for n, p in [
    ("you_should", r"\byou\s+(?:should|may\s+want\s+to|might\s+want\s+to|could\s+consider|ought\s+to)\s+(?:consider\s+)?(?:buy|buying|sell|selling|invest|investing|purchase|hold|holding|short|accumulate)\b"),
    ("good_investment", r"\b(?:is|are|remains?|looks?|seems?)\s+(?:a\s+|an\s+)?(?:good|great|solid|strong|attractive|compelling|excellent|smart)\s+(?:investment|buy|stock|opportunity|pick)\b"),
    ("i_recommend", r"\b(?:i|we)\s+(?:would\s+)?(?:recommend|suggest|advise)\s+(?:buying|selling|investing|holding|purchasing|that\s+you)\b"),
    ("rating", r"\b(?:strong\s+buy|strong\s+sell|buy\s+rating|sell\s+rating|outperform\s+rating|price\s+target\s+of)\b"),
    ("prediction", r"\b(?:the\s+)?(?:stock|shares|share\s+price)\s+(?:will|is\s+likely\s+to|is\s+expected\s+to)\s+(?:rise|go\s+up|increase|double|outperform|fall|drop|decline)\b"),
    ("undervalued", r"\b(?:is|are|appears?|looks?)\s+(?:currently\s+)?(?:undervalued|overvalued|a\s+bargain)\b"),
]]


def match_rules(text: str, rules: list[tuple[str, re.Pattern]]) -> list[str]:
    norm = normalize(text)
    return [name for name, rx in rules if rx.search(norm)]


class AdviceGuard(BaseGuard):
    name = "advice"
    stage = "input"
    fail_action = "block"

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        hits = match_rules(ctx.question, INPUT_RULES)
        return not hits, 1.0 if hits else 0.0, {"rules": hits}


class AdviceOutputGuard(BaseGuard):
    name = "advice_output"
    stage = "output"
    fail_action = "block"

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        hits = match_rules(ctx.answer or "", OUTPUT_RULES)
        return not hits, 1.0 if hits else 0.0, {"rules": hits}
