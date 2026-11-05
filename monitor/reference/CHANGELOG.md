# Reference window changelog

The drift reference is refreshed deliberately, never automatically; otherwise drift gets absorbed
silently. Every refresh is a new file (`vN.npz`) plus an entry here, approved by the maintainer.

| Version | Date | Contents | Why |
|---|---|---|---|
| v1 | 2026-11-05 | 74 golden questions + 420 simulated baseline requests (7 days x 60, seed 100). Scope and retrieval-confidence score distributions under guards/v2. | Initial reference. Production traffic is too low; the simulator's baseline mix stands in for "the first weeks of traffic". Replace with real traffic once there are a few hundred production requests. |

Refresh with:

```bash
python -m monitor.reference build --env production --start YYYY-MM-DD --end YYYY-MM-DD --version v2 --out monitor/reference/v2.npz
```

then point `reference:` in `configs/monitoring.yaml` at the new file in the same PR.
