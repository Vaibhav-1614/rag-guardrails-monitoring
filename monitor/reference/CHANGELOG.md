# Reference window changelog

The drift reference is refreshed deliberately, never automatically; otherwise drift gets absorbed
silently. Every refresh is a new file (`vN.npz`) plus an entry here, approved by the maintainer.

| Version | Date | Contents | Why |
|---|---|---|---|
| v1 | 2026-11-05 | 74 golden questions + 420 simulated baseline requests (7 days x 60, seed 100). Scope and retrieval-confidence score distributions under guards/v2. | Initial reference. Production traffic is too low; the simulator's baseline mix stands in for "the first weeks of traffic". Replace with real traffic once there are a few hundred production requests. |
| v2 | 2026-11-15 | 176 upstream golden questions + 420 simulated baseline requests (7 days x 60, seed 100). Score distributions under the upstream `v2-bm25` pipeline and guards/v2. | The service moved onto the upstream v2 pipeline: the retrieval-confidence signal changed from a lexical/cosine relevance score to upstream's coverage/evidence score, so v1's distributions no longer apply. v1 removed. |

Refresh with:

```bash
python -m monitor.reference build --env production --start YYYY-MM-DD --end YYYY-MM-DD --version v3 --out monitor/reference/v3.npz
```

then point `reference:` in `configs/monitoring.yaml` at the new file in the same PR.
