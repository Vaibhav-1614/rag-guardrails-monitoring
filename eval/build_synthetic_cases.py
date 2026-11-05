"""Build labelled synthetic cases for the two deterministic output guards.

eval/redteam/numeric_grounding.jsonl
    (answer, cited chunk, label) triples built from real 10-K table rows and sentences. Supported
    answers copy, re-scale ("416,161" in millions -> "$416.2 billion") or correctly derive a figure;
    unsupported answers corrupt it (wrong digit, wrong scale, a 0.3-0.9% near miss, a 3-15% shift,
    wrong percentage) or attach a real figure from another row ("attribution" - a known blind spot
    of presence-based checking). Labels are correct by construction: every corrupted value is checked
    to be absent from the chunk with an independent exact/0.05% comparison, not with the guard.

eval/redteam/citation.jsonl
    Extractive answers to golden questions (valid citations) and corrupted copies: out-of-range [n],
    all citations removed, one sentence's citation removed.

    python eval/build_synthetic_cases.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rag.companies import get_registry  # noqa: E402
from rag.corpus import load_corpus  # noqa: E402
from rag.llm import ExtractiveGenerator, GenerationRequest  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

ROW = re.compile(r"^(?P<label>[A-Z][A-Za-z ,&'’()\-]{3,60}?)\s+\$?\s?(?P<v1>\(?\d{1,3}(?:,\d{3})+\)?)\s+\$?\s?(?P<v2>\(?\d{1,3}(?:,\d{3})+\)?)", re.M)
PCT_SENT = re.compile(r"[^.\n]{20,220}?\b(?P<pct>\d{1,3}(?:\.\d)?)%[^.\n]{0,160}\.")
NUM_ANY = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _num(s: str) -> float:
    return float(s.strip("()").replace(",", ""))


def _values_in(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUM_ANY.findall(text) if m.replace(",", "").replace(".", "").isdigit()]


def absent(value: float, text: str, scales=(1.0, 1e3, 1e6, 1e9)) -> bool:
    """Independent label check: value differs from every chunk number (at any table scale) by > 0.05%."""
    for v in _values_in(text):
        for s in scales:
            if v and abs(v * s - value) <= 0.0005 * abs(value):
                return False
    return True


def fmt_billions(x: float, decimals: int) -> str:
    return f"${x / 1e9:,.{decimals}f} billion"


def numeric_cases(rng: random.Random) -> list[dict]:
    corpus = load_corpus()
    reg = get_registry()
    rows_by_ticker: dict[str, list] = {}
    sents_by_ticker: dict[str, list] = {}
    for c in corpus.chunks:
        for m in ROW.finditer(c.text):
            label = m.group("label").strip(" ,")
            if len(label.split()) > 7 or re.search(r"\d", label):
                continue
            rows_by_ticker.setdefault(c.ticker, []).append((c, label, m.group("v1"), m.group("v2")))
        for m in PCT_SENT.finditer(c.text):
            sents_by_ticker.setdefault(c.ticker, []).append((c, m.group(0).strip(), m.group("pct")))
    out: list[dict] = []
    sup_cycle = ["exact", "with_unit", "rounded_billion", "rounded_coarse", "derived_change"]
    bad_cycle = ["digit_change", "scale_error", "near_miss", "shift", "attribution", "wrong_derived"]
    k = 0
    for t in reg.tickers:
        name = reg.name(t)
        rows = rows_by_ticker.get(t, [])
        rng.shuffle(rows)
        scale = corpus.table_scale.get(t, 1e6)
        used = 0
        for c, label, v1, v2 in rows:
            if used >= 7:
                break
            a, b = _num(v1), _num(v2)
            if a < 100 or b <= 0 or "(" in v1 or "(" in v2:
                continue
            used += 1
            sup = sup_cycle[(k) % len(sup_cycle)]
            bad = bad_cycle[(k) % len(bad_cycle)]
            k += 1
            lab = label[0].lower() + label[1:] if not label.isupper() else label.title()
            # ---- supported variant
            if sup == "exact":
                ans, target = f"{name} reported {lab} of {v1} [1].", v1
            elif sup == "with_unit" and scale == 1e6:
                ans, target = f"{name} reported {lab} of ${v1} million [1].", f"${v1} million"
            elif sup in ("rounded_billion", "with_unit") and a * scale >= 1e9:
                target = fmt_billions(a * scale, 1)
                ans = f"{name}'s {lab} was about {target} [1]."
            elif sup == "rounded_coarse" and a * scale >= 1e10:
                target = fmt_billions(a * scale, 0)
                ans = f"{name}'s {lab} was roughly {target} [1]."
            else:
                pct = abs(a - b) / b * 100
                target = f"{pct:.1f}%"
                ans = f"{name}'s {lab} changed by {target} between the two years shown, from {v2} to {v1} [1]."
                sup = "derived_change"
            out.append({"id": f"num{len(out) + 1:03d}", "category": sup, "label": "supported", "answer": ans,
                        "target": target, "chunk_ids": [c.chunk_id], "ticker": t})
            # ---- unsupported variant
            text = c.text
            if bad == "digit_change":
                digits = [i for i, ch in enumerate(v1) if ch.isdigit()]
                for _ in range(10):
                    i = rng.choice(digits[1:]) if len(digits) > 1 else digits[0]
                    new = v1[:i] + str((int(v1[i]) + rng.randint(1, 8)) % 10) + v1[i + 1:]
                    if absent(_num(new), text):
                        break
                ans, target = f"{name} reported {lab} of {new} [1].", new
            elif bad == "scale_error" and scale == 1e6:
                ans, target = f"{name} reported {lab} of ${v1} billion [1].", f"${v1} billion"
            elif bad in ("near_miss", "scale_error"):
                for _ in range(20):
                    f = 1 + rng.choice([-1, 1]) * rng.uniform(0.003, 0.009)
                    target = fmt_billions(a * scale * f, 2) if a * scale >= 1e9 else f"{a * f:,.0f}"
                    val = a * scale * f if a * scale >= 1e9 else round(a * f)
                    if absent(val, text) and abs(val - a * (scale if a * scale >= 1e9 else 1)) / (a * (scale if a * scale >= 1e9 else 1)) >= 0.003:
                        break
                ans = f"{name}'s {lab} was {target} [1]."
                bad = "near_miss"
            elif bad == "shift":
                for _ in range(20):
                    f = 1 + rng.choice([-1, 1]) * rng.uniform(0.03, 0.15)
                    target = f"{a * f:,.0f}"
                    if absent(round(a * f), text):
                        break
                ans = f"{name} reported {lab} of {target} [1]."
            elif bad == "attribution":
                others = [m for m in ROW.finditer(text) if m.group("label").strip(" ,") != label and _num(m.group("v1")) >= 100
                          and absent(_num(m.group("v1")), f"{label} {v1} {v2}")]
                if not others:
                    continue
                o = rng.choice(others)
                target = o.group("v1")
                ans = f"{name} reported {lab} of {target} [1]."
            else:  # wrong_derived
                pct = abs(a - b) / b * 100
                wrong = pct + rng.choice([-1, 1]) * rng.uniform(3, 8)
                if wrong <= 0.5:
                    wrong = pct + rng.uniform(3, 8)
                target = f"{wrong:.1f}%"
                ans = f"{name}'s {lab} changed by {target} between the two years shown, from {v2} to {v1} [1]."
            out.append({"id": f"num{len(out) + 1:03d}", "category": bad, "label": "unsupported", "answer": ans,
                        "target": target, "chunk_ids": [c.chunk_id], "ticker": t})
        # ---- percentage sentences from prose
        sents = sents_by_ticker.get(t, [])
        rng.shuffle(sents)
        for c, sent, pct in sents[:3]:
            if len(NUM_ANY.findall(sent)) > 3:
                continue
            out.append({"id": f"num{len(out) + 1:03d}", "category": "exact_sentence", "label": "supported",
                        "answer": f"{sent[:-1]} [1].", "target": f"{pct}%", "chunk_ids": [c.chunk_id], "ticker": t})
            p = float(pct)
            for _ in range(10):
                wrong = round(p + rng.choice([-1, 1]) * rng.choice([2, 3, 5, 7, 9]), 1 if "." in pct else 0)
                if wrong > 0 and f"{wrong:g}%" not in c.text:
                    break
            wrong_s = f"{wrong:g}%"
            bad_sent = sent[:-1].replace(f"{pct}%", wrong_s, 1)
            out.append({"id": f"num{len(out) + 1:03d}", "category": "wrong_percent", "label": "unsupported",
                        "answer": f"{bad_sent} [1].", "target": wrong_s, "chunk_ids": [c.chunk_id], "ticker": t})
    return out


def citation_cases(rng: random.Random) -> list[dict]:
    golden = [json.loads(line) for line in (ROOT / "eval/golden.jsonl").read_text().splitlines() if line.strip()]
    retriever = Retriever()
    gen = ExtractiveGenerator()
    out: list[dict] = []
    for g in golden:
        if not g["answerable"]:
            continue
        r = retriever.retrieve(g["question"])
        chunks = [{"ticker": c.ticker, "section": c.section, "text": c.text, "chunk_id": c.chunk_id} for c in r.chunks]
        ans = gen.generate(GenerationRequest(g["question"], chunks, "", "")).text
        if "[" not in ans:
            continue
        ids = [c["chunk_id"] for c in chunks]
        base = {"question": g["question"], "chunk_ids": ids, "golden_id": g["id"]}
        variant = len(out) % 4
        if variant == 0:
            out.append({**base, "category": "valid", "label": "valid", "answer": ans})
        elif variant == 1:
            bad = re.sub(r"\[(\d+)\]", f"[{len(ids) + rng.randint(1, 4)}]", ans, count=1)
            out.append({**base, "category": "out_of_range", "label": "invalid", "answer": bad})
        elif variant == 2:
            out.append({**base, "category": "no_citations", "label": "invalid", "answer": re.sub(r"\s*\[\d+\]", "", ans)})
        else:
            sents = re.split(r"(?<=\]\.)\s+", ans)
            if len(sents) < 2:
                out.append({**base, "category": "valid", "label": "valid", "answer": ans})
                continue
            i = rng.randrange(len(sents))
            sents[i] = re.sub(r"\s*\[\d+\]", "", sents[i])
            out.append({**base, "category": "one_uncited", "label": "invalid", "answer": " ".join(sents)})
    for i, row in enumerate(out, 1):
        row["id"] = f"cit{i:03d}"
    return out


def main() -> None:
    rng = random.Random(7)
    num = numeric_cases(rng)
    (ROOT / "eval/redteam/numeric_grounding.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in num))
    cit = citation_cases(random.Random(11))
    (ROOT / "eval/redteam/citation.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cit))
    from collections import Counter

    print("numeric:", len(num), Counter((r["label"], r["category"]) for r in num))
    print("citation:", len(cit), Counter(r["category"] for r in cit))


if __name__ == "__main__":
    main()
