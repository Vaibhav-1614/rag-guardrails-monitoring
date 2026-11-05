# Guardrails and Drift Monitoring for a SEC 10-K RAG Service

Input and output guardrails, a red-team evaluation suite with CI gates, and nightly drift monitoring for a
retrieval-augmented Q&A service over the latest 10-K filings of 10 large-cap companies (AAPL, MSFT, GOOGL, AMZN,
META, NVDA, TSLA, JPM, JNJ, XOM). Implements [design doc 02](docs/02-guardrails-and-monitoring-design.md).
Runs at **$0/month**: guards run on CPU inside the service, monitoring runs as a scheduled GitHub Actions job.

The corpus (5,769 chunks of 256 tokens, with section tags) is vendored from
[rag-eval-platform](https://github.com/Vaibhav-1614/rag-eval-platform), which benchmarked retrieval on it. This
repo adds the serving layer (FastAPI, tracing, golden set, CI) that the design assumes from doc 01, then the
guardrails and monitoring on top.

![Ask page: PII redacted before retrieval and generation, every guard's verdict shown](docs/assets/dash_ask.png)

## Results

All numbers below were produced by `make eval` and `make drift-validate` in this repo, in **lite mode**: rules +
Presidio patterns + LSA embeddings + lexical groundedness + an offline extractive answer generator (no LLM key, no
model downloads). The model-backed backends (bge-small, a DeBERTa prompt-injection classifier, an NLI
cross-encoder, spaCy NER) are implemented and wired into config but **were not measured** here; see
[Limitations](#limitations-read-these).

### Guards (config `guards/v2`, 471 labelled red-team and synthetic cases + 74 golden questions)

| Guard | Precision | Recall | Golden-set false positives | Set |
|---|---|---|---|---|
| Prompt injection (rules) | 1.00 | 1.00 (62/62) | 0 / 66 | 62 attacks, 22 near-miss benign |
| ↳ *held-out test split, tuned on dev only* | *1.00* | ***0.83 (25/30)*** | *0* | [snapshot](eval/results/injection_heldout_snapshot.json) |
| PII redaction | 1.00 | 1.00 (17/17 values) | 0 / 66 | 16 positives, 12 tricky negatives (exec names, CIKs, corporate phones) |
| Scope (off-topic + unknown company) | 1.00 | 1.00 | 0 / 66 (+0 / 12 near-miss) | 22 positives |
| Investment-advice intent | 1.00 | 1.00 | 0 / 66 (+0 / 12 near-miss) | 14 positives |
| Advice language in answers | 1.00 | 1.00 | – | 12 positives, 8 filing-style negatives |
| **Numeric grounding** | **0.99** | **0.98** (0.86 incl. attribution errors) | – | 186 synthetic answer/chunk pairs |
| Citation validator | 1.00 | 1.00 | – | 65 cases |
| Retrieval-confidence gate | – | 0.88 of unanswerable, 1.00 of off-topic | 0 / 66 | 8 unanswerable, 12 off-topic |
| Context sanitizer (indirect injection) | – | 10/10 planted, payload removed 10/10 | 0 / 5,769 real chunks | 10 synthetic chunks |
| NLI groundedness (lexical fallback, shadow mode) | 1.00 | 1.00 | – | 130 grounded / swapped-context pairs (easy) |

**System level:** 0 answerable golden questions blocked · injection attack success rate 0% · guard-caused
over-refusal 0% on 100 legitimate questions · PII masked in every stored trace (tested) · guard overhead **24 ms p50 / 48 ms p95**
per request on CPU (budget: 500 ms).

![Precision-recall curves with the chosen operating points](docs/assets/pr_curves.png)

### The numbers that matter most

1. **The doc's ±0.5% numeric tolerance is too loose.** At 0.5% the numeric guard misses wrong-digit figures such as
   "$418.2B" when the filing says $416.2B: recall 0.83. Precision-aware matching does better. A figure counts as
   supported if the filing value rounds or truncates to it at the precision the answer shows, so "$416.2 billion" and
   "$416 billion" are both fine for 416,161 (in millions). With only 0.05% extra slack it gets recall 0.98 at
   precision 0.99. v2 ships that.

   ![Numeric grounding vs tolerance](docs/assets/numeric_tolerance.png)

2. **Presence is not attribution.** Recall on the 11 "attribution" cases is 0/11. These attach a *real* figure from
   another row of the same table to the wrong line item. A presence-based check can't catch that; NLI or an LLM judge
   has to. The headline recall excludes them, and the gate is on that number, but they stay in the set.

3. **Held-out generalisation of the injection rules: 0.83, not 1.00.** Rules were tuned on the dev split only, then
   scored once on the held-out test split (25/30, 0 false positives). The five test misses (base64 payloads, "ignore
   the filings and make up numbers", secret-listing requests, ...) were fixed afterwards, so the final 1.00 is an
   upper bound on hand-written data. Expect lower on real attacks.

4. **Shadow mode caught real problems before enforcement.** The v1 config, the doc's starting thresholds, fails 2
   of 13 gates on the same code. The retrieval gate at 0.45 refused an answerable golden question (min answerable
   score: 0.434). The numeric guard at 0.5% tolerance had recall 0.83. The first eval run also found a retrieval bug:
   zero-score BM25 ties padded the candidate list and earned RRF credit. A sanitizer rule also flagged 8 real 10-K
   chunks ("without limitation" is legal boilerplate). All were fixed before promotion; v2 passes every gate.

5. **The regenerate loop works (simulated).** Using a generator that corrupts 30% of the figures it writes:

   | Numeric guard mode | Answers with figures that reach users with corrupted, unflagged numbers | Flagged | Regenerated |
   |---|---|---|---|
   | monitor (shadow) | 64.5% | 0% | 0% |
   | enforce (regenerate once, then flag) | **0%** | 32.3% | 24.2% |

   This exercises the pipeline mechanics with an artificial error rate. It is not a measurement of any real LLM.

### Rollout: v1 (shadow) → v2 (promoted)

Every guard started where [v1](configs/guards/v1.yaml) put it. [v2](configs/guards/v2.yaml) promotes a guard only
after its false-positive rate was measured at zero on the 66 answerable golden questions:

| Guard | v1 | v2 | Evidence |
|---|---|---|---|
| prompt_injection | monitor, 0.90 | **enforce**, 0.90 | 0 FP on golden + 22 near-miss; rules no longer hit any real 10-K chunk |
| scope | monitor, 0.35 | **enforce**, 0.45 (+ unknown-company sub-check) | legit min score 0.54 vs off-topic max 0.39: threshold in the gap |
| retrieval_conf | enforce, 0.45 | enforce, **0.42** | answerable min 0.434; v1 blocked one |
| numeric_grounding | monitor, 0.5% | **enforce**, precision-aware + 0.05% | tolerance sweep above; action is regenerate-then-flag, never block |
| citation | enforce | enforce, strict claims | the doc's number/entity heuristic missed uncited prose sentences |
| context_sanitizer | enforce | enforce, quarantine whole chunk | sentence stripping left a payload's second sentence behind |
| nli_grounding | monitor | **monitor** | only the lexical fallback was measurable offline; stays shadow |

### Drift monitoring, validated on simulated traffic

A portfolio service doesn't get enough real traffic for drift statistics to mean anything. **Drift detectors are
validated against simulated traffic scenarios; production traffic is low.** Each scenario simulates 14 days at 60
requests/day: 7 baseline days, then the scenario. The nightly job runs on each scenario night the way the cron would.

| Scenario | Onset | Expected detector fired (night) | Also fired | False alarms before onset |
|---|---|---|---|---|
| baseline | – | – | – | **0 in 7 nightly runs** |
| topic shift (companies outside the corpus) | day 7 | scope drift (7), query drift (13) | output, guard (7), emerging topics (11) | 0 |
| attack wave (35% injections on one day) | day 13 | guard drift: injection trigger rate (13) | output, scope (13) | 0 |
| hallucination pressure (bait questions) | day 7 | output drift: refusal rate (9) | retrieval drift (13) | 0 |
| upstream model drift (silent degradation) | day 9 | golden-run drift (9), output drift (9) | guard, ops (9) | 0 |

Details: [docs/drift_validation.md](docs/drift_validation.md). Two design fixes came out of validation:
- Bands use median/MAD, not mean/σ, so a few drifted days can't widen the band and absorb the shift.
- Rate metrics are floored at their binomial standard error. Without that floor, sampling noise at 60 requests/day raised two pre-onset alarms.

Query drift on topic shift fires only on night 13, with AUC 0.71 against a 0.70 threshold. That is a known weakness
of the LSA embedder: words it never saw in the 10-Ks ("Netflix") contribute nothing. The scope detector catches the
shift on night 7.

![Scenario time series](docs/assets/drift_scenarios.png)

## Architecture

```mermaid
flowchart TD
    Q[Question] --> IG
    subgraph IG["Input guards (parallel)"]
        I1[Length / charset] --- I2[PII redact] --- I3[Injection rules + classifier] --- I4[Scope / unknown company] --- I5[Advice intent]
    end
    IG -->|block| REF1[Policy / scope / advice refusal]
    IG -->|pass, redacted| RET[Hybrid retrieval: BM25 + dense, weighted RRF]
    RET --> RC{"Retrieval confidence >= 0.42?"}
    RC -- no --> REF2["Not found in the filings"]
    RC -- yes --> SAN["Context sanitizer: delimit, escape, quarantine"]
    SAN --> GEN["LLM: Gemini / Groq, or offline extractive"]
    GEN --> OG
    subgraph OG["Output guards"]
        O1[Citations] --- O2[Numeric grounding] --- O3[NLI groundedness] --- O4[Advice language] --- O5[PII]
    end
    OG -->|fail, 1st time| REGEN[Regenerate with stricter prompt] --> OG
    OG -->|fail again| FB["Strip uncited / flag / safe fallback"]
    OG -->|pass| RESP[Response]
    OG -. masked trace .-> LF[("Langfuse or local JSONL")]
    LF --> NIGHT["Nightly GitHub Action: drift job + golden run"]
    NIGHT --> REP["Report on monitoring branch, GitHub issue, Discord"]
    NIGHT --> RQ["Review queue -> eval/*.jsonl"]
```

Every guard implements one interface (`check(ctx) -> GuardResult` with `guard`, `passed`, `score`, `action`,
`detail`, `latency_ms`). Each runs in `monitor`, `enforce` or `off` mode from YAML, and each per-guard result is
stored on the trace.

### What each guard does (and where it deviates from the doc)

- **Charset** (§5.1): length > 1,000, no letters, zero-width / bidi / Unicode-tag characters, homoglyph words that
  mix Latin with Cyrillic or Greek.
- **PII** (§5.2): Microsoft Presidio, fully offline. `tldextract` is pinned to its bundled suffix list, so Presidio
  never phones home. Redaction happens before retrieval, the LLM and tracing. The tracer masks every string again,
  so PII never leaves the process whatever the guard modes are. The allowlist is built from the corpus: 115 signatories
  from the signature pages, company names, and corporate phone numbers from cover pages. PERSON entities come from
  spaCy NER when `en_core_web_sm` is installed; otherwise a case-sensitive self-identification pattern ("my name is …")
  covers the common case. A real bug found here: Presidio compiles patterns case-insensitively by default.
- **Prompt injection** (§5.3): 39 weighted rule families scanned over normalised variants of the text
  (NFKC, invisible characters stripped, spaced-out letters joined, leetspeak, decoded base64 payloads). The rule
  score is `1 − Π(1 − wᵢ)`, a graded score with a PR curve. An optional HF classifier score is logged separately;
  `block if rule_score ≥ t or classifier ≥ t`.
- **Scope** (§5.4): `0.55·domain + 0.30·similarity + 0.15·mentions-covered-company`. *domain* is the corpus-vs-English
  word log-likelihood ratio (`wordfreq`); *similarity* is the max cosine to (ticker, section) centroids. Max-cosine to
  single chunks was tried and rejected: "write me a poem about the ocean" scored 0.79 under LSA.
  **Unknown company** is a deterministic sub-check with its own mode, so it can be enforced while the fuzzy score is
  still in shadow. Subsidiaries (YouTube, AWS, LinkedIn) route to the parent.
- **Advice** (§5.4/§6.6): intent rules on the question, recommendation-language rules on the answer. Questions about
  what a filing *says* about investing risk stay allowed.
- **Retrieval confidence** (§6.1): RRF has no absolute meaning, so each chunk gets an absolute relevance score:
  `0.6 × stemmed IDF-weighted query-term coverage + 0.4 × cosine`. Clauses that carried redacted PII ("My email is
  <EMAIL>, …") are dropped from the search query. Before that fix, their words alone pushed legitimate questions
  below the gate.
- **Context sanitizer** (§6.2): chunks are wrapped in `<chunk id ticker section>` with `<`/`>` escaped, so a chunk
  can't forge a delimiter. The system prompt says chunk text is data. Any chunk that trips the injection rules is
  quarantined whole.
- **Citations** (§6.3): `[n]` must exist; every substantive sentence needs a citation ("strict" mode; the doc's
  number/entity heuristic is available as `claims: heuristic`). Second failure → uncited sentences stripped, or safe
  fallback if citations are invalid.
- **Numeric grounding** (§6.4): extracts currency, percent, bps, years and plain figures with scale words or
  letters, and handles parentheses negatives and table units ("in millions" per chunk, else the filing's dominant
  unit). It skips identifiers (Item 7, Note 12, 10-K, 401(k), H100, Q3, COVID-19, day-of-month). Figures must come
  from the **cited** chunk; one found only in another chunk is "miscited". Derivations are verified only when the
  sentence states the operation (percent change, difference, sum, share), from figures in the same sentence or the
  same table row. Years inside the filing's three reported fiscal years are accepted; others must be in the context.
- **NLI groundedness** (§6.5): a cross-encoder over sentence windows of the cited chunks (`nli-deberta-v3-xsmall`),
  with a lexical-overlap fallback. Stays in shadow mode.

### Drift detectors (§8)

| Detector | Signal | Method | Fires when |
|---|---|---|---|
| Query drift | query embeddings, 7-day window vs. reference | classifier two-sample test (logistic regression, out-of-fold AUC), RBF-MMD permutation test, centroid distance | AUC ≥ 0.70 and p < 0.001 |
| Emerging topics | query embeddings | queries farther from the reference than 95% of reference points are from each other, clustered with k-means, labelled by TF-IDF terms | ≥ 15% novel and a cluster ≥ 10 queries |
| Scope drift | scope scores, out-of-scope rate | PSI vs. reference + robust daily band | PSI > 0.2 and ≥ 8 pt move, or outside band |
| Retrieval drift | top relevance score | PSI vs. reference | PSI > 0.2 |
| Output drift | refusal, regeneration, flagged, unsupported-number rates; citations; answer length; groundedness | median ± 3·MAD band over the trailing 7 days, binomial floor for rates | outside band and ≥ min change |
| Guard drift | per-guard trigger rate | binomial z-test vs. trailing 7 days | z ≥ 4, ≥ 2× baseline, ≥ 5 hits |
| Upstream model drift | nightly golden-set run against production | band over previous runs; says whether the code/config version changed | outside band |
| Ops drift | p95 latency, error / fallback rate, tokens per request | daily band | outside band |

Every detector has a minimum sample size, and thresholds live in [configs/monitoring.yaml](configs/monitoring.yaml).
The reference window is versioned ([monitor/reference/](monitor/reference/CHANGELOG.md)) and refreshed only by hand.

**Feedback loop (§9).** `python -m monitor.feedback export` writes flagged traces to a review queue: guard failures,
low groundedness, thumbs-down from `POST /feedback`, and members of emerging-topic clusters. `--langfuse-queue` also
pushes them to a Langfuse annotation queue. After labelling, `apply` appends:
- true issues to the matching red-team set;
- false positives and new question types to the golden set.

Entries are tagged `source: production:<trace_id>`, so `feedback stats` reports eval-set growth from production.
Raw PII can never enter the eval sets: traces are masked before storage.

## Limitations (read these)

- **Only lite backends were measured.** This build environment couldn't reach the Hugging Face Hub, so the
  bge / DeBERTa / NLI / spaCy paths are untested code. They are covered by the same interfaces, and the `ml-guards` CI
  job runs them on demand. Thresholds for bge need recalibrating (`by_embedder` overrides exist), and the classifier
  benchmark from the doc's open question 1 is still to do.
- **No LLM in the loop.** Without API keys, answers come from an extractive generator that copies sentences
  verbatim, so the golden unsupported-number rate (0%) and hallucination-bait results say nothing about a real model's
  hallucination rate. The `generation-eval` CI job runs those sets with Gemini/Groq when keys are configured.
- **Hand-written, single-annotator red-team data**, and the rules were developed against it. Treat the 1.00 scores as
  upper bounds; the held-out 0.83 is the honest generalisation estimate. Public prompt-injection datasets are not
  included yet.
- **Small sets.** 8 unanswerable golden questions and a 0.434-vs-0.42 margin on the retrieval gate; 7 legitimate
  near-miss questions are refused by that gate. I didn't verify they're answerable from the filings.
- **Langfuse and GitHub integrations are tested against mocked HTTP**, not live accounts. The Docker image layout was
  smoke-tested, but `docker build` wasn't run (no daemon here).
- Retrieval quality is not the focus here: hybrid Hit@5 = 0.96 on the 50 prose golden questions (lite mode).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements-dev.txt          # add requirements-ml.txt for the model-backed guards
```

```bash
make test                                    # 63 tests
```

```bash
make eval                                    # guard eval + CI gates -> eval/results/, docs/assets/
```

```bash
make drift-validate                          # ~4 min: simulate scenarios, check every detector
```

```bash
make serve                                   # FastAPI on :8000 (POST /ask, POST /feedback, GET /health, GET /config)
```

```bash
make dashboard                               # Streamlit: Ask, Guard evaluation, Monitoring
```

Copy `.env.example` to `.env` for LLM keys (Gemini, Groq), Langfuse, the simulator secret and alert webhooks. Without
keys everything still runs: extractive answers, and traces as JSONL in `data/traces/`. Any setting can be overridden
with `RAG__SECTION__KEY`. For example, `RAG__GUARDS_CONFIG=configs/guards/v1.yaml` serves the shadow-mode config.

**CI** (`.github/workflows/ci.yml`):
- Every PR: lint, tests and guard gates (no LLM calls).
- Generation-dependent sets: only when prompt, guard or config files change and a key exists.
- Drift validation: whenever monitoring code changes.

**Nightly** (`monitor.yml`):
- Runs the golden set against `MONITOR_TARGET_URL` and the drift job over Langfuse traces.
- Commits the report to the `monitoring` branch and opens or updates one GitHub issue per firing detector.

## Repository layout

```
src/rag/        corpus, retriever, embedder (bge | lsa), llm (gemini | groq | extractive | fault-injecting),
                prompts, tracing (langfuse | local, PII-masked), service orchestrator, FastAPI app
src/guards/     base, pipeline (modes, parallel input guards, regenerate loop), charset, pii, injection, scope,
                advice, retrieval_conf, sanitizer, citation, numeric, nli
configs/        settings.yaml, guards/v1.yaml (shadow rollout), guards/v2.yaml (promoted), monitoring.yaml,
                companies.yaml, pii_allowlist.txt (generated)
eval/           golden.jsonl, redteam/*.jsonl, run_guard_eval.py, gates.yaml, build_synthetic_cases.py, results/
monitor/        run.py (nightly job), drift.py (stats), golden_run.py, simulate_traffic.py, validate_detectors.py,
                reference.py + reference/v1.npz, alerts.py, feedback.py, validation/ (sample reports)
dashboard/      app.py + pages/guard_evaluation.py, pages/monitoring.py
data/index/     vendored 10-K chunks (+ section tags) and bge embeddings, MANIFEST.json with provenance
```

## Portfolio summary

- Designed a configurable guardrail pipeline (PII redaction, prompt-injection detection, scope/advice policy,
  citation and numeric-grounding checks) with shadow-mode rollout and gate-backed promotion. Precision-aware numeric
  grounding catches 98% of corrupted financial figures at 99% precision, versus 83% for the commonly suggested ±0.5%
  tolerance. It was promoted with 0 false blocks on the golden set.
- Built a red-team evaluation suite: 220 hand-written adversarial and near-miss cases plus 251 synthetic grounding
  cases, with per-guard precision/recall gates in CI. Prompt-injection attack success is 0% on the full set. The
  held-out estimate is 83% recall at 0 false positives, after tuning on a separate dev split.
- Implemented nightly drift monitoring: classifier two-sample and MMD tests on query embeddings, PSI on retrieval and
  scope scores, robust bands on output behaviour, guard-rate spikes, and upstream-model regression via nightly golden
  runs. Alerts go to GitHub issues and a webhook. Every detector was validated on five simulated scenarios with 0
  false alarms, and a feedback loop turns flagged traces into new eval cases.

## License

MIT. Filing text is public SEC EDGAR data; see `data/index/MANIFEST.json` for the source filings.
