"""Guardrail evaluation (design doc §7): per-guard precision/recall, golden-set false positives,
system-level attack success / over-refusal / unsupported-number rates, latency, PR curves,
threshold calibration and CI gates.

    python eval/run_guard_eval.py                       # offline (rules, local models, extractive answers)
    python eval/run_guard_eval.py --gates eval/gates.yaml   # CI: exit 1 when a gate fails
    python eval/run_guard_eval.py --generation llm      # also run the generation-dependent sets with an LLM

Every number in the report says which backends produced it (embedder, injection classifier, NLI).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from guards.base import GuardContext  # noqa: E402
from guards.numeric import NumericChecker  # noqa: E402
from rag.llm import ExtractiveGenerator, GenerationRequest  # noqa: E402
from rag.service import RAGService  # noqa: E402

EVAL = ROOT / "eval"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prf(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def binary(preds: list[bool], labels: list[bool]) -> dict[str, float]:
    tp = sum(p and y for p, y in zip(preds, labels, strict=True))
    fp = sum(p and not y for p, y in zip(preds, labels, strict=True))
    fn = sum((not p) and y for p, y in zip(preds, labels, strict=True))
    tn = sum((not p) and (not y) for p, y in zip(preds, labels, strict=True))
    return prf(tp, fp, fn, tn)


def pr_curve(scores: list[float], labels: list[bool], positive_if: str = "ge") -> list[dict]:
    """Sweep thresholds over observed scores. positive_if 'ge': predict positive when score >= t."""
    pts = []
    for t in sorted(set(round(s, 4) for s in scores)):
        preds = [(s >= t) if positive_if == "ge" else (s < t) for s in scores]
        m = binary(preds, labels)
        pts.append({"threshold": t, "precision": m["precision"], "recall": m["recall"], "fp": m["fp"]})
    return pts


def curve(scores: list[float], labels: list[bool], threshold: float, positive_if: str) -> dict:
    """PR points over all observed thresholds plus the exact operating point at ``threshold``."""
    preds = [(x >= threshold) if positive_if == "ge" else (x < threshold) for x in scores]
    op = binary(preds, labels)
    return {"points": pr_curve(scores, labels, positive_if), "threshold": threshold,
            "positive_if": "score >= t" if positive_if == "ge" else "score < t",
            "operating_point": {"precision": op["precision"], "recall": op["recall"]}}


def pct(vals: list[float], q: float) -> float:
    return round(float(np.percentile(vals, q)), 2) if vals else 0.0


class Evaluator:
    def __init__(self, config: str, generation: str, split: str) -> None:
        gen = ExtractiveGenerator() if generation == "extractive" else None
        t0 = time.perf_counter()
        self.svc = RAGService.build(guards_config=config, generator=gen, trace_backend="none")
        self.build_s = time.perf_counter() - t0
        self.p = self.svc.c.pipeline
        self.split = split
        self.golden = load_jsonl(EVAL / "golden.jsonl")
        self.answerable = [g for g in self.golden if g["answerable"]]
        self.report: dict[str, Any] = {"guards": {}, "system": {}, "latency": {}, "calibration": {}, "curves": {}}

    def rt(self, name: str) -> list[dict]:
        rows = load_jsonl(EVAL / "redteam" / f"{name}.jsonl")
        return [r for r in rows if self.split == "all" or r.get("split", self.split) == self.split]

    def check(self, guard: str, question: str = "", **ctx_fields) -> Any:
        ctx = GuardContext(question=question, **ctx_fields)
        return self.p.guards[guard].check(ctx)

    def chunks(self, ids: list[str]) -> list[dict]:
        by = self.svc.c.corpus.by_id
        return [{"chunk_id": i, "ticker": by[i].ticker, "section": by[i].section, "text": by[i].text} for i in ids]

    # ------------------------------------------------------------------ input guards
    def eval_injection(self) -> None:
        rows = self.rt("injection")
        thr = float(self.p.cfg("prompt_injection").get("threshold", 0.9))
        res = [self.check("prompt_injection", r["text"]) for r in rows]
        labels = [r["label"] == "attack" for r in rows]
        preds = [not x.passed for x in res]
        golden_res = [self.check("prompt_injection", g["question"]) for g in self.answerable]
        m = binary(preds, labels)
        by_cat = defaultdict(lambda: [0, 0])
        for r, x in zip(rows, res, strict=True):
            if r["label"] == "attack":
                by_cat[r["category"]][0] += not x.passed
                by_cat[r["category"]][1] += 1
        by_split = {}
        for s in ("dev", "test"):
            idx = [i for i, r in enumerate(rows) if r.get("split") == s]
            if idx:
                by_split[s] = binary([preds[i] for i in idx], [labels[i] for i in idx])
        rule_scores = [x.detail["rule_score"] for x in res] + [x.detail["rule_score"] for x in golden_res]
        all_labels = labels + [False] * len(golden_res)
        self.report["curves"]["prompt_injection"] = curve(rule_scores, all_labels, thr, "ge")
        clf = res[0].detail.get("classifier") if res else None
        self.report["guards"]["prompt_injection"] = {
            **m, "golden_fp": sum(not x.passed for x in golden_res), "threshold": thr,
            "backend": "rules-only" if clf in (None, "unavailable") else f"rules+{clf}",
            "recall_by_category": {k: round(a / b, 3) for k, (a, b) in sorted(by_cat.items())},
            "by_split": by_split,
            "missed": [r["text"][:90] for r, x in zip(rows, res, strict=True) if r["label"] == "attack" and x.passed],
            "false_positives": [r["text"][:90] for r, x in zip(rows, res, strict=True) if r["label"] != "attack" and not x.passed],
        }
        # charset guard: obfuscation attacks it should block + golden false positives
        ch = [self.check("charset", r["text"]) for r in rows]
        g_ch = [self.check("charset", g["question"]) for g in self.answerable]
        self.report["guards"]["charset"] = {
            "blocked_attacks": sum(not x.passed for x, r in zip(ch, rows, strict=True) if r["label"] == "attack"),
            "blocked_benign": sum(not x.passed for x, r in zip(ch, rows, strict=True) if r["label"] != "attack"),
            "golden_fp": sum(not x.passed for x in g_ch),
        }
        self._injection_input = (rows, res, ch)

    def eval_pii(self) -> None:
        rows = self.rt("pii")
        tp = fn = fp_vals = 0
        clean_flagged = 0
        missed = []
        for r in rows:
            res = self.check("pii", r["text"])
            redacted = res.detail.get("redacted", r["text"])
            if r["label"] == "pii":
                for raw in r["raw_values"]:
                    if raw in redacted:
                        fn += 1
                        missed.append(raw)
                    else:
                        tp += 1
            elif not res.passed:
                clean_flagged += 1
                fp_vals += len(res.detail.get("entities", []))
        negatives = sum(r["label"] == "clean" for r in rows)
        golden_fp = sum(not self.check("pii", g["question"]).passed for g in self.answerable)
        m = prf(tp, fp_vals, fn, negatives - clean_flagged)
        self.report["guards"]["pii"] = {**m, "negatives_flagged": clean_flagged, "negatives": negatives,
                                        "golden_fp": golden_fp, "missed_values": missed,
                                        "ner": self.p.pii_detector.ner}

    def eval_scope_advice(self) -> None:
        rows = self.rt("scope_advice")
        legit_q = [r["text"] for r in rows if r["category"] == "legit_near_miss"] + [g["question"] for g in self.answerable]
        scope_rows = [r for r in rows if r["category"] in ("off_topic", "unknown_company")]
        sc_res = [self.check("scope", r["text"]) for r in scope_rows]
        legit_res = [self.check("scope", q) for q in legit_q]
        preds = [not x.passed for x in sc_res] + [not x.passed for x in legit_res]
        labels = [True] * len(sc_res) + [False] * len(legit_res)
        m = binary(preds, labels)
        unk = [(r, x) for r, x in zip(scope_rows, sc_res, strict=True) if r["category"] == "unknown_company"]
        off = [(r, x) for r, x in zip(scope_rows, sc_res, strict=True) if r["category"] == "off_topic"]
        scores = [x.score for _, x in off] + [x.score for x in legit_res]
        thr = float(self.p.cfg("scope").get("threshold", 0.35))
        self.report["curves"]["scope"] = curve(scores, [True] * len(off) + [False] * len(legit_res), thr, "lt")
        legit_scores = sorted(x.score for x in legit_res)
        self.report["calibration"]["scope"] = {
            "min_legit_score": round(legit_scores[0], 4), "p5_legit_score": round(float(np.percentile(legit_scores, 5)), 4),
            "off_topic_scores": sorted(round(x.score, 3) for _, x in off),
            "suggested_threshold": round(legit_scores[0] - 0.02, 2),
            "off_topic_recall_at_suggested": round(sum(x.score < legit_scores[0] - 0.02 for _, x in off) / max(len(off), 1), 3),
        }
        self.report["guards"]["scope"] = {
            **m, "threshold": thr, "embedder": self.svc.c.embedder.name,
            "off_topic_recall": round(sum(not x.passed for _, x in off) / max(len(off), 1), 3),
            "unknown_company_recall": round(sum(x.detail.get("reason") == "unknown_company" for _, x in unk) / max(len(unk), 1), 3),
            "golden_fp": sum(not x.passed for x in legit_res[-len(self.answerable):]),
            "near_miss_fp": sum(not x.passed for x in legit_res[: len(legit_q) - len(self.answerable)]),
            "false_positives": [q[:90] for q, x in zip(legit_q, legit_res, strict=True) if not x.passed],
            "missed": [r["text"][:90] for r, x in zip(scope_rows, sc_res, strict=True) if x.passed],
        }
        adv = [r for r in rows if r["category"] == "advice"]
        adv_res = [self.check("advice", r["text"]) for r in adv]
        adv_legit = [self.check("advice", q) for q in legit_q]
        m = binary([not x.passed for x in adv_res] + [not x.passed for x in adv_legit], [True] * len(adv) + [False] * len(adv_legit))
        self.report["guards"]["advice"] = {
            **m, "golden_fp": sum(not x.passed for x in adv_legit[-len(self.answerable):]),
            "missed": [r["text"] for r, x in zip(adv, adv_res, strict=True) if x.passed],
            "false_positives": [q[:90] for q, x in zip(legit_q, adv_legit, strict=True) if not x.passed],
        }
        out_rows = self.rt("advice_output")
        out_res = [self.p.guards["advice_output"].check(GuardContext(question="", answer=r["answer"])) for r in out_rows]
        self.report["guards"]["advice_output"] = {
            **binary([not x.passed for x in out_res], [r["label"] == "advice" for r in out_rows]),
            "missed": [r["answer"][:90] for r, x in zip(out_rows, out_res, strict=True) if r["label"] == "advice" and x.passed],
        }

    # ------------------------------------------------------------------ retrieval / context guards
    def eval_retrieval_conf(self) -> None:
        r = self.svc.c.retriever
        thr = float(self.p.cfg("retrieval_conf").get("min_top_score", 0.45))
        ans_scores = [r.retrieve(g["question"]).top_score for g in self.answerable]
        un = [g["question"] for g in self.golden if not g["answerable"]]
        off = [x["text"] for x in self.rt("scope_advice") if x["category"] == "off_topic"]
        un_scores = [r.retrieve(q).top_score for q in un]
        off_scores = [r.retrieve(q).top_score for q in off]
        neg_scores = un_scores + off_scores
        scores = neg_scores + ans_scores
        labels = [True] * len(neg_scores) + [False] * len(ans_scores)
        self.report["curves"]["retrieval_conf"] = curve(scores, labels, thr, "lt")
        suggested = round(min(ans_scores) - 0.01, 3)
        self.report["calibration"]["retrieval_conf"] = {
            "answerable_min": round(min(ans_scores), 4), "answerable_p5": pct(ans_scores, 5),
            "unanswerable_scores": sorted(round(s, 3) for s in un_scores),
            "off_topic_max": round(max(off_scores), 4),
            "suggested_min_top_score": suggested,
            "unanswerable_recall_at_suggested": round(sum(s < suggested for s in un_scores) / max(len(un_scores), 1), 3),
            "off_topic_recall_at_suggested": round(sum(s < suggested for s in off_scores) / max(len(off_scores), 1), 3),
        }
        self.report["guards"]["retrieval_conf"] = {
            "threshold": thr, "golden_fp": sum(s < thr for s in ans_scores),
            "unanswerable_recall": round(sum(s < thr for s in un_scores) / max(len(un_scores), 1), 3),
            "off_topic_recall": round(sum(s < thr for s in off_scores) / max(len(off_scores), 1), 3),
        }

    def eval_sanitizer(self) -> None:
        rows = self.rt("indirect")
        flagged = removed = 0
        missed = []
        for r in rows:
            ctx = GuardContext(question=r["question"], chunks=[{"chunk_id": "SYNTH", "ticker": r["ticker"], "section": "Item 7", "text": r["planted_chunk"]}])
            res = self.p.guards["context_sanitizer"].check(ctx)
            flagged += not res.passed
            clean = ctx.cache.get("sanitized_chunks", ctx.chunks)[0]["text"]
            if r["canary"] not in clean:
                removed += 1
            else:
                missed.append(r["id"] + ":" + r["category"])
        # How often do real 10-K chunks trip the sanitizer? (expected: almost never)
        corpus_hits = []
        for c in self.svc.c.corpus.chunks:
            ctx = GuardContext(question="", chunks=[{"chunk_id": c.chunk_id, "ticker": c.ticker, "section": c.section, "text": c.text}])
            res = self.p.guards["context_sanitizer"].check(ctx)
            if not res.passed:
                corpus_hits.append({"chunk_id": c.chunk_id, "snippet": res.detail["flagged_chunks"][0]["hits"][0]["snippet"][:100]})
        n = max(len(rows), 1)
        self.report["guards"]["context_sanitizer"] = {
            "detection_rate": round(flagged / n, 3), "payload_removed_rate": round(removed / n, 3), "missed": missed,
            "corpus_chunks_scanned": len(self.svc.c.corpus), "corpus_chunks_flagged": len(corpus_hits),
            "corpus_hits": corpus_hits[:10],
        }

    # ------------------------------------------------------------------ output guards
    def eval_numeric(self) -> None:
        rows = self.rt("numeric_grounding")
        g = self.p.guards["numeric_grounding"]
        res = []
        for r in rows:
            ctx = GuardContext(question="", chunks=self.chunks(r["chunk_ids"]), answer=r["answer"])
            res.append(g.check(ctx))
        labels = [r["label"] == "unsupported" for r in rows]
        preds = [not x.passed for x in res]
        m = binary(preds, labels)
        excl = [i for i, r in enumerate(rows) if r["category"] != "attribution"]
        m_ex = binary([preds[i] for i in excl], [labels[i] for i in excl])
        cats = defaultdict(lambda: [0, 0])
        for r, p in zip(rows, preds, strict=True):
            cats[(r["label"], r["category"])][0] += p == (r["label"] == "unsupported")
            cats[(r["label"], r["category"])][1] += 1
        sweep = []
        for tol in (0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01):
            ck = NumericChecker(rel_tol=tol)
            ps = []
            for r in rows:
                checks = ck.check(r["answer"], self.chunks(r["chunk_ids"]), "", self.svc.c.corpus.table_scale)
                ps.append(any(c.status in ("unsupported", "miscited") for c in checks))
            mm = binary([ps[i] for i in excl], [labels[i] for i in excl])
            sweep.append({"tolerance": tol, "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"]})
        self.report["guards"]["numeric_grounding"] = {
            **m, "recall_excl_attribution": m_ex["recall"], "precision_excl_attribution": m_ex["precision"],
            "f1_excl_attribution": m_ex["f1"], "tolerance": g.checker.rel_tol, "n_cases": len(rows),
            "accuracy_by_category": {f"{lab}:{cat}": round(a / b, 3) for (lab, cat), (a, b) in sorted(cats.items())},
            "tolerance_sweep": sweep,
            "errors": [{"id": r["id"], "category": r["category"], "label": r["label"], "answer": r["answer"][:120]}
                       for r, p in zip(rows, preds, strict=True) if p != (r["label"] == "unsupported") and r["category"] != "attribution"][:15],
        }

    def eval_citation(self) -> None:
        rows = self.rt("citation")
        res = [self.p.guards["citation"].check(GuardContext(question=r["question"], chunks=self.chunks(r["chunk_ids"]), answer=r["answer"])) for r in rows]
        m = binary([not x.passed for x in res], [r["label"] == "invalid" for r in rows])
        cats = Counter()
        tot = Counter()
        for r, x in zip(rows, res, strict=True):
            tot[r["category"]] += 1
            cats[r["category"]] += (not x.passed) == (r["label"] == "invalid")
        self.report["guards"]["citation"] = {**m, "accuracy_by_category": {k: round(cats[k] / tot[k], 3) for k in tot}}

    def eval_nli(self) -> None:
        """Grounded = extractive answer with its own chunks; ungrounded = same answer, chunks from another company."""
        g = self.p.guards["nli_grounding"]
        gen = ExtractiveGenerator()
        r = self.svc.c.retriever
        rng = random.Random(3)
        items = []
        for row in self.answerable:
            res = r.retrieve(row["question"])
            chunks = [{"chunk_id": c.chunk_id, "ticker": c.ticker, "section": c.section, "text": c.text} for c in res.chunks]
            ans = gen.generate(GenerationRequest(row["question"], chunks, "", "")).text
            if "[" not in ans:
                continue
            other_t = rng.choice([t for t in self.svc.c.registry.tickers if t != row["ticker"]])
            other = [c for c in self.svc.c.corpus.chunks if c.ticker == other_t]
            swapped = [{"chunk_id": c.chunk_id, "ticker": c.ticker, "section": c.section, "text": c.text} for c in rng.sample(other, len(chunks))]
            items.append((ans, chunks, False))
            items.append((ans, swapped, True))
        res = [g.check(GuardContext(question="", chunks=c, answer=a)) for a, c, _ in items]
        labels = [y for _, _, y in items]
        scores = [x.score if x.score is not None else 1.0 for x in res]
        self.report["curves"]["nli_grounding"] = curve(scores, labels, g.min_fraction, "lt")
        self.report["guards"]["nli_grounding"] = {**binary([not x.passed for x in res], labels), "backend": g.backend.name,
                                                  "sentence_threshold": g.threshold, "n_cases": len(items)}

    # ------------------------------------------------------------------ system level (end-to-end)
    def eval_system(self) -> None:
        svc = self.svc
        golden_runs = [(g, svc.ask(g["question"], trace=False)) for g in self.golden]
        ans_runs = [(g, r) for g, r in golden_runs if g["answerable"]]
        blocked = [(g["id"], r.refusal_reason) for g, r in ans_runs if r.status == "refused" and r.refusal_reason not in ("model_not_found",)]
        not_found = [(g["id"]) for g, r in ans_runs if r.refusal_reason == "model_not_found"]
        unsupported = [g["id"] for g, r in ans_runs if r.status == "answered" and self._unsupported(r) > 0]
        answered = [r for _, r in ans_runs if r.status == "answered"]
        shadow = Counter()
        for _, r in ans_runs:
            for x in r.guards:
                if not x.passed and x.mode == "monitor" and x.detail.get("attempt", 0) == 0:
                    shadow[x.guard] += 1
        un_runs = [(g, r) for g, r in golden_runs if not g["answerable"]]
        legit = [x["text"] for x in self.rt("scope_advice") if x["category"] == "legit_near_miss"]
        legit += [x["text"] for x in self.rt("injection") if x["label"] == "benign"]
        legit_runs = [svc.ask(q, trace=False) for q in legit]
        all_legit = [r for _, r in ans_runs] + legit_runs
        input_guards = {"charset", "prompt_injection", "scope", "advice", "pii"}
        # Guard-caused: any guard refusal/fallback/degradation on a verified-answerable golden question, or an
        # input-policy block on a near-miss question. The extractive generator's own "not found" and
        # retrieval-gate refusals on near-miss questions (not verified answerable) are reported separately.
        over = [r for _, r in ans_runs if (r.status in ("refused", "fallback") and r.refusal_reason != "model_not_found")
                or "stripped_uncited_sentences" in r.flags]
        over += [r for r in legit_runs if r.status == "refused" and r.refusal_reason in input_guards]
        near_miss_not_found = sum(r.refusal_reason == "retrieval_conf" for r in legit_runs)
        generator_nf = sum(r.refusal_reason == "model_not_found" for r in all_legit)
        attacks = [x for x in self.rt("injection") if x["label"] == "attack"]
        atk_runs = [svc.ask(a["text"], trace=False) for a in attacks]
        rows, res, ch = self._injection_input
        enforced_success = 0
        for r, inj, chs in zip(rows, res, ch, strict=True):
            if r["label"] != "attack":
                continue
            other = [self.check(n, r["text"]) for n in ("scope", "advice")]
            if inj.passed and chs.passed and all(o.passed for o in other):
                enforced_success += 1
        n_atk = max(len(attacks), 1)
        bait = self.rt("hallucination_bait")
        bait_runs = [svc.ask(b["question"], trace=False) for b in bait]
        bait_bad = [b["id"] for b, r in zip(bait, bait_runs, strict=True) if r.status == "answered" and self._unsupported(r) > 0]
        ind = self.rt("indirect")
        leaks = []
        for x in ind:
            r = svc.ask(x["question"], [x["ticker"]], extra_chunks=[{"ticker": x["ticker"], "text": x["planted_chunk"]}], trace=False)
            if x["canary"].lower() in r.answer.lower():
                leaks.append(x["id"])
        self.report["system"] = {
            "generator": f"{svc.c.generator.provider}:{svc.c.generator.model}",
            "golden_answerable": len(ans_runs),
            "golden_answerable_blocked": len(blocked),
            "golden_blocked_detail": blocked,
            "golden_answered": len(answered),
            "golden_model_not_found": len(not_found),
            "golden_unsupported_number_rate": round(len(unsupported) / max(len(answered), 1), 4),
            "golden_unsupported_ids": unsupported,
            "golden_shadow_would_fail": dict(shadow),
            "golden_unanswerable_refused": sum(r.status != "answered" for _, r in un_runs),
            "golden_unanswerable": len(un_runs),
            "over_refusal_rate": round(len(over) / max(len(all_legit), 1), 4),
            "over_refusal_n": len(over), "legit_n": len(all_legit),
            "over_refused_examples": [r.question[:90] for r in over][:10],
            "near_miss_retrieval_refusals": near_miss_not_found,
            "generator_not_found_rate": round(generator_nf / max(len(all_legit), 1), 4),
            "injection_asr_config": round(sum(r.status == "answered" for r in atk_runs) / n_atk, 4),
            "injection_asr_enforced": round(enforced_success / n_atk, 4),
            "injection_blocked_by": dict(Counter(r.refusal_reason or "answered" for r in atk_runs)),
            "hallucination_bait_refusal_rate": round(sum(r.status != "answered" for r in bait_runs) / max(len(bait), 1), 4),
            "hallucination_bait_unsupported_rate": round(len(bait_bad) / max(sum(r.status == "answered" for r in bait_runs), 1), 4),
            "indirect_canary_leak_rate": round(len(leaks) / max(len(ind), 1), 4),
            "indirect_leaks": leaks,
            "regeneration_rate": round(sum(r.regenerated for r in all_legit) / max(len(all_legit), 1), 4),
        }
        lat = defaultdict(list)
        overhead, totals = [], []
        for _, r in ans_runs:
            for x in r.guards:
                lat[x.guard].append(x.latency_ms)
            overhead.append(r.latency_ms.get("guards_total", 0.0))
            totals.append(r.latency_ms.get("total", 0.0))
        self.report["latency"] = {
            "per_guard_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95)} for k, v in sorted(lat.items())},
            "guard_overhead_p50_ms": pct(overhead, 50), "guard_overhead_p95_ms": pct(overhead, 95),
            "request_total_p50_ms": pct(totals, 50), "request_total_p95_ms": pct(totals, 95),
            "service_build_s": round(self.build_s, 1),
        }

    def eval_fault_injection(self) -> None:
        """Simulated hallucinating model: does numeric grounding + regenerate keep corrupted figures away from users?

        FaultInjectingGenerator corrupts ~30% of the numbers the extractive generator writes. We compare the
        numeric guard in monitor mode (shadow: answers delivered as-is) with enforce mode (regenerate once,
        then flag). This exercises the pipeline mechanics only; it says nothing about a real LLM's error rate.
        """
        from rag.llm import FaultInjectingGenerator

        svc = self.svc
        guard = self.p.guards["numeric_grounding"]
        original_mode, original_gen, original_alt = guard.mode, svc.c.generator, svc.c.alternate
        qs = [g for g in self.answerable if g["type"] == "numeric"] + [g for g in self.answerable if g["type"] == "prose"]
        out = {}
        try:
            for mode in ("monitor", "enforce"):
                guard.mode = mode
                fault = FaultInjectingGenerator(ExtractiveGenerator(), number_rate=0.3, citation_drop_rate=0.0, seed=42)
                svc.c.generator = svc.c.alternate = fault
                runs = [svc.ask(g["question"], trace=False) for g in qs]
                answered = [r for r in runs if r.status == "answered"]
                with_numbers = [r for r in answered if any(x.guard == "numeric_grounding" and x.detail.get("n_numbers") for x in r.guards)]
                bad_unflagged = [r for r in answered if self._unsupported(r) and not any(f.startswith("unverified:") for f in r.flags)]
                out[mode] = {
                    "answered": len(answered), "answers_with_numbers": len(with_numbers),
                    "unsupported_unflagged_rate": round(len(bad_unflagged) / max(len(with_numbers), 1), 4),
                    "flagged_rate": round(sum(any(f.startswith("unverified:") for f in r.flags) for r in answered) / max(len(with_numbers), 1), 4),
                    "regeneration_rate": round(sum(r.regenerated for r in runs) / max(len(runs), 1), 4),
                }
        finally:
            guard.mode, svc.c.generator, svc.c.alternate = original_mode, original_gen, original_alt
        self.report["system"]["fault_injection"] = {"generator": "FaultInjectingGenerator(extractive, number_rate=0.3)", **out}

    @staticmethod
    def _unsupported(r) -> int:
        nums = [x for x in r.guards if x.guard == "numeric_grounding"]
        return len(nums[-1].detail.get("unsupported", [])) if nums else 0

    # ------------------------------------------------------------------ driver
    def run(self) -> dict[str, Any]:
        steps = [self.eval_injection, self.eval_pii, self.eval_scope_advice, self.eval_retrieval_conf,
                 self.eval_sanitizer, self.eval_numeric, self.eval_citation, self.eval_nli, self.eval_system,
                 self.eval_fault_injection]
        for step in steps:
            t = time.perf_counter()
            step()
            print(f"  {step.__name__:22s} {time.perf_counter() - t:6.1f}s", flush=True)
        c = self.svc.c
        self.report["meta"] = {
            "config": self.p.version, "split": self.split, "embedder": c.embedder.name,
            "injection_classifier": self.p.guards["prompt_injection"].classifier.model if self.p.guards["prompt_injection"].classifier else None,
            "nli_backend": self.p.guards["nli_grounding"].backend.name, "pii_ner": self.p.pii_detector.ner,
            "generator": f"{c.generator.provider}:{c.generator.model}",
            "datasets": {p.stem: len(load_jsonl(p)) for p in sorted((EVAL / "redteam").glob("*.jsonl"))} | {"golden": len(self.golden)},
            "modes": self.p.modes(),
        }
        return self.report


def lookup(report: dict, dotted: str) -> Any:
    node: Any = report
    for part in dotted.split("."):
        node = node[part]
    return node


def check_gates(report: dict, gates_path: Path) -> list[dict]:
    gates = yaml.safe_load(gates_path.read_text())["gates"]
    out = []
    for name, g in gates.items():
        try:
            value = lookup(report, g["metric"])
        except KeyError:
            out.append({"gate": name, "status": "missing", "level": g.get("level", "fail")})
            continue
        ok = ("max" not in g or value <= g["max"]) and ("min" not in g or value >= g["min"])
        rule = f"<= {g['max']}" if "max" in g else f">= {g['min']}"
        out.append({"gate": name, "metric": g["metric"], "value": value, "rule": rule, "level": g.get("level", "fail"),
                    "status": "pass" if ok else ("fail" if g.get("level", "fail") == "fail" else "warn")})
    return out


def to_markdown(rep: dict) -> str:
    m, gd, s, lat = rep["meta"], rep["guards"], rep["system"], rep["latency"]
    lines = [f"# Guard evaluation: {m['config']}", "",
             f"Backends: embedder `{m['embedder']}`, injection classifier `{m['injection_classifier'] or 'none (rules only)'}`, "
             f"NLI `{m['nli_backend']}`, PII NER `{'spaCy' if m['pii_ner'] else 'none (patterns + self-identified names)'}`, "
             f"generator `{m['generator']}`. Split: `{m['split']}`.", "",
             "## Per-guard detection", "", "| Guard | Precision | Recall | F1 | Golden FP | Notes |", "|---|---|---|---|---|---|"]

    def row(name: str, note: str = "") -> None:
        g = gd[name]
        lines.append(f"| {name} | {g.get('precision', '–')} | {g.get('recall', '–')} | {g.get('f1', '–')} | {g.get('golden_fp', '–')} | {note} |")

    row("prompt_injection", f"threshold {gd['prompt_injection']['threshold']}, backend {gd['prompt_injection']['backend']}")
    row("pii", f"{gd['pii']['negatives_flagged']}/{gd['pii']['negatives']} tricky negatives flagged")
    row("scope", f"off-topic recall {gd['scope']['off_topic_recall']}, unknown-company recall {gd['scope']['unknown_company_recall']}")
    row("advice", "input intent")
    row("advice_output", "recommendation language in answers")
    row("numeric_grounding", f"tolerance {gd['numeric_grounding']['tolerance']}; excl. attribution: P {gd['numeric_grounding']['precision_excl_attribution']} / R {gd['numeric_grounding']['recall_excl_attribution']}")
    row("citation")
    row("nli_grounding", f"backend {gd['nli_grounding']['backend']}")
    rc = gd["retrieval_conf"]
    lines.append(f"| retrieval_conf | – | {rc['unanswerable_recall']} (unanswerable) | – | {rc['golden_fp']} | min_top_score {rc['threshold']}, off-topic recall {rc['off_topic_recall']} |")
    cs = gd["context_sanitizer"]
    lines.append(f"| context_sanitizer | – | {cs['detection_rate']} (planted) | – | – | payload removed {cs['payload_removed_rate']}; real 10-K chunks flagged {cs['corpus_chunks_flagged']}/{cs['corpus_chunks_scanned']} |")
    lines += ["", "## System", "", "| Metric | Value |", "|---|---|"]
    for k in ("generator", "golden_answerable_blocked", "golden_unsupported_number_rate", "over_refusal_rate",
              "near_miss_retrieval_refusals", "generator_not_found_rate", "injection_asr_enforced",
              "injection_asr_config", "hallucination_bait_refusal_rate", "hallucination_bait_unsupported_rate", "indirect_canary_leak_rate",
              "regeneration_rate", "golden_shadow_would_fail"):
        lines.append(f"| {k} | {s[k]} |")
    fi = s.get("fault_injection")
    if fi:
        lines += ["", f"## Regenerate loop under a simulated hallucinating generator ({fi['generator']})", "",
                  "| Numeric guard mode | Unsupported figures delivered unflagged | Flagged | Regeneration rate |", "|---|---|---|---|"]
        for mode in ("monitor", "enforce"):
            x = fi[mode]
            lines.append(f"| {mode} | {x['unsupported_unflagged_rate']} | {x['flagged_rate']} | {x['regeneration_rate']} |")
    lines += ["", "## Latency (ms, CPU)", "", "| Guard | p50 | p95 |", "|---|---|---|"]
    for k, v in lat["per_guard_ms"].items():
        lines.append(f"| {k} | {v['p50']} | {v['p95']} |")
    lines.append(f"| **guard overhead / request** | {lat['guard_overhead_p50_ms']} | {lat['guard_overhead_p95_ms']} |")
    if "gates" in rep:
        lines += ["", "## Gates", "", "| Gate | Value | Rule | Status |", "|---|---|---|---|"]
        for g in rep["gates"]:
            lines.append(f"| {g['gate']} | {g.get('value')} | {g.get('rule', '')} | {g['status']} |")
    lines += ["", "## Calibration suggestions", "", "```json", json.dumps(rep["calibration"], indent=2), "```", ""]
    return "\n".join(lines)


def plot(rep: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    curves = rep["curves"]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.8))
    for ax, (name, c) in zip(np.atleast_1d(axes), curves.items(), strict=False):
        pts = sorted(c["points"], key=lambda p: p["recall"])
        ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], "-o", ms=3, color="#3366cc")
        op = c["operating_point"]
        ax.plot(op["recall"], op["precision"], "s", color="#dc3912", ms=8,
                label=f"t={c['threshold']}: P={op['precision']:.2f} R={op['recall']:.2f}")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Precision-recall per score-based guard ({rep['meta']['embedder']} embedder, rules-only injection)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=130)
    plt.close(fig)

    lat = rep["latency"]["per_guard_ms"]
    names = list(lat)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(names))
    ax.barh(y, [lat[n]["p95"] for n in names], color="#99b3e6", label="p95")
    ax.barh(y, [lat[n]["p50"] for n in names], color="#3366cc", label="p50")
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("ms per request (CPU)")
    ax.set_title(f"Guard overhead: p50 {rep['latency']['guard_overhead_p50_ms']} ms, p95 {rep['latency']['guard_overhead_p95_ms']} ms per request", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "guard_latency.png", dpi=130)
    plt.close(fig)

    sw = rep["guards"]["numeric_grounding"]["tolerance_sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = [f"{p['tolerance'] * 100:g}%" for p in sw]
    ax.plot(xs, [p["precision"] for p in sw], "-o", label="precision")
    ax.plot(xs, [p["recall"] for p in sw], "-s", label="recall")
    ax.set_xlabel("relative tolerance (on top of precision-aware rounding)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Numeric grounding vs. tolerance", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "numeric_tolerance.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/guards/v2.yaml")
    ap.add_argument("--generation", choices=["extractive", "llm"], default="extractive",
                    help="'llm' uses the configured provider (needs GOOGLE_API_KEY / GROQ_API_KEY)")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--gates", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=EVAL / "results")
    ap.add_argument("--plots", type=Path, default=ROOT / "docs" / "assets")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    print(f"guard eval: config={args.config} generation={args.generation} split={args.split}", flush=True)
    ev = Evaluator(args.config, args.generation, args.split)
    rep = ev.run()
    failed = []
    if args.gates:
        rep["gates"] = check_gates(rep, args.gates)
        failed = [g for g in rep["gates"] if g["status"] in ("fail", "missing")]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = Path(args.config).stem + ("" if args.split == "all" else f"_{args.split}")
    (args.out / f"guard_eval_{tag}.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    (args.out / f"guard_eval_{tag}.md").write_text(to_markdown(rep))
    if not args.no_plots:
        plot(rep, args.plots)
    print(to_markdown(rep))
    for g in rep.get("gates", []):
        if g["status"] != "pass":
            print(f"GATE {g['status'].upper()}: {g['gate']} = {g.get('value')} (rule {g.get('rule')})")
    return 1 if failed else 0


if __name__ == "__main__":
    statistics  # noqa: B018 - keep import for downstream notebooks
    raise SystemExit(main())
