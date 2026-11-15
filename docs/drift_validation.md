# Drift detector validation

Simulated traffic: 14 days x 60 requests/day per scenario (days 0-6 baseline, scenario from day 7; model drift degrades the model from day 9), nightly job run on each of days 7-13 as the cron would; cells show the first night a detector fired. Reference: golden set (176) + 7 days of simulated baseline traffic (420 requests, seed 100, 2026-09-01..2026-09-07). Embedder: `lsa`.

| Scenario | Onset | query_drift | emerging_topics | scope_drift | retrieval_drift | output_drift | guard_drift | upstream_model_drift | ops_drift | False alarms before onset | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | – | · | · | · | · | · | · | · | · | none | yes |
| topic_shift | day 7 | **fired** (day 13) | fired (day 12) | **fired** (day 7) | · | fired (day 7) | fired (day 7) | · | · | none | yes |
| attack_wave | day 13 | · | · | fired (day 13) | · | fired (day 13) | **fired** (day 13) | · | · | none | yes |
| hallucination_pressure | day 7 | · | · | · | · | **fired** (day 7) | fired (day 7) | · | · | none | yes |
| model_drift | day 9 | · | · | · | · | **fired** (day 9) | fired (day 9) | **fired** (day 9) | fired (day 9) | none | yes |

Baseline: **0** false alarms over 7 consecutive nightly runs.

**fired** = expected detector fired; fired = additional detector fired (secondary signal); MISSED = expected detector stayed quiet; · = quiet.

- **topic_shift**: scope_drift/out_of_scope_rate=0.0571; output_drift/refusal_rate=0.2; guard_drift/scope_trigger_rate=0.1667; emerging_topics/novel_share=0.1881; query_drift/c2st_auc=0.7106
- **attack_wave**: scope_drift/out_of_scope_rate=0.0667; output_drift/refusal_rate=0.3833; guard_drift/prompt_injection_trigger_rate=0.3333; guard_drift/scope_trigger_rate=0.25
- **hallucination_pressure**: output_drift/refusal_rate=0.2333; guard_drift/retrieval_conf_trigger_rate=0.2
- **model_drift**: output_drift/refusal_rate=0.2833; output_drift/regeneration_rate=0.6667; output_drift/flagged_rate=0.3167; output_drift/unsupported_rate=0.2791; guard_drift/citation_trigger_rate=0.25; guard_drift/numeric_grounding_trigger_rate=0.3167
