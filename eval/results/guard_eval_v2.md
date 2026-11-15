# Guard evaluation: guards/v2

Backends: embedder `lsa`, injection classifier `none (rules only)`, NLI `lexical`, PII NER `none (patterns + self-identified names)`, generator `extractive:extractive-v2`, upstream retrieval `v2-bm25` (re-ranker off) at `8c2c8c7d6a0a`. Split: `all`.

## Per-guard detection

| Guard | Precision | Recall | F1 | Golden FP | Notes |
|---|---|---|---|---|---|
| prompt_injection | 1.0 | 1.0 | 1.0 | 0 | threshold 0.9, backend rules-only |
| pii | 1.0 | 1.0 | 1.0 | 0 | 0/12 tricky negatives flagged |
| scope | 1.0 | 1.0 | 1.0 | 0 | off-topic recall 1.0, unknown-company recall 1.0 |
| advice | 1.0 | 1.0 | 1.0 | 0 | input intent |
| advice_output | 1.0 | 1.0 | 1.0 | – | recommendation language in answers |
| numeric_grounding | 0.9867 | 0.881 | 0.9308 | – | tolerance 0.0005; excl. attribution: P 0.9867 / R 0.9737 |
| citation | 1.0 | 0.9694 | 0.9845 | – |  |
| nli_grounding | 0.9737 | 1.0 | 0.9867 | – | backend lexical |
| retrieval_conf | – | 0.857 (unanswerable) | – | 0 | upstream `v2-bm25`, re-ranker off; hallucination-bait recall 0.633 |
| context_sanitizer | – | 1.0 (planted) | – | – | payload removed 1.0; real 10-K chunks flagged 0/7061 |

## System

| Metric | Value |
|---|---|
| generator | extractive:extractive-v2 |
| golden_answerable | 148 |
| golden_answerable_blocked | 0 |
| golden_unanswerable_refusal_rate | 0.9286 |
| golden_answer_check | 0.8971 |
| golden_by_split | {'dev': {'answerable': 105, 'answerable_refused_by_guards': 0, 'unanswerable': 20, 'unanswerable_refused': 19, 'answer_check': 0.9412, 'answer_check_n': 34}, 'test': {'answerable': 43, 'answerable_refused_by_guards': 0, 'unanswerable': 8, 'unanswerable_refused': 7, 'answer_check': 0.8529, 'answer_check_n': 34}} |
| golden_unsupported_number_rate | 0.0 |
| over_refusal_rate | 0.0 |
| near_miss_retrieval_refusals | 2 |
| generator_not_found_rate | 0.0 |
| injection_asr_enforced | 0.0 |
| injection_asr_config | 0.0 |
| hallucination_bait_refusal_rate | 0.6333 |
| hallucination_bait_unsupported_rate | 0.0 |
| indirect_canary_leak_rate | 0.0 |
| regeneration_rate | 0.0 |
| golden_shadow_would_fail | {} |

## Regenerate loop under a simulated hallucinating generator (FaultInjectingGenerator(extractive, number_rate=0.3))

| Numeric guard mode | Unsupported figures delivered unflagged | Flagged | Regeneration rate |
|---|---|---|---|
| monitor | 0.5811 | 0.0 | 0.0 |
| enforce | 0.0 | 0.3919 | 0.371 |

## Latency (ms, CPU)

| Guard | p50 | p95 |
|---|---|---|
| advice | 0.07 | 0.11 |
| advice_output | 0.11 | 0.29 |
| charset | 0.1 | 0.17 |
| citation | 0.21 | 0.5 |
| context_sanitizer | 8.2 | 19.16 |
| nli_grounding | 0.37 | 1.01 |
| numeric_grounding | 1.39 | 3.12 |
| pii | 2.15 | 3.25 |
| pii_output | 1.96 | 3.89 |
| prompt_injection | 0.16 | 0.25 |
| retrieval_conf | 0.01 | 0.02 |
| scope | 6.08 | 9.45 |
| **guard overhead / request** | 21.71 | 34.4 |

## Gates

| Gate | Value | Rule | Status |
|---|---|---|---|
| golden_false_blocks | 0 | <= 0 | pass |
| injection_asr | 0.0 | <= 0.1 | pass |
| injection_golden_fp | 0 | <= 0 | pass |
| advice_refusal_recall | 1.0 | >= 0.95 | pass |
| pii_redaction_recall | 1.0 | >= 0.95 | pass |
| pii_golden_fp | 0 | <= 0 | pass |
| numeric_recall | 0.9737 | >= 0.9 | pass |
| numeric_precision | 0.9867 | >= 0.9 | pass |
| citation_recall | 0.9694 | >= 0.95 | pass |
| unsupported_number_rate | 0.0 | <= 0.05 | pass |
| over_refusal_rate | 0.0 | <= 0.1 | pass |
| indirect_payload_removed | 1.0 | >= 0.8 | pass |
| guard_overhead_p95_ms | 34.4 | <= 500 | pass |

## Calibration suggestions

```json
{
  "scope": {
    "min_legit_score": 0.5549,
    "p5_legit_score": 0.636,
    "off_topic_scores": [
      0.012,
      0.069,
      0.099,
      0.112,
      0.138,
      0.155,
      0.159,
      0.191,
      0.259,
      0.387,
      0.389,
      0.389
    ],
    "suggested_threshold": 0.53,
    "off_topic_recall_at_suggested": 1.0
  },
  "retrieval_conf": {
    "signal": "question-term coverage (no re-ranker loaded)",
    "dev_answerable_min_confidence": 0.333,
    "dev_answerable_p5_confidence": 0.63,
    "unanswerable_confidences": [
      0.0,
      0.2,
      0.333,
      0.429,
      0.5,
      0.5,
      0.5,
      0.5,
      0.5,
      0.5,
      0.6,
      0.6,
      0.6,
      0.667,
      0.667,
      0.667,
      0.667,
      0.75,
      0.75,
      0.75,
      0.8,
      0.833,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  }
}
```
