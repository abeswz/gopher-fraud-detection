# PROGRESS

**Last updated:** 2026-06-05 (CPU redistrib — HAProxy bottleneck eliminated)

## Status

Detection perfeita (FP:0, FN:0) local e remoto. **Score local = 6000 (MAX)** após redistribuição de CPU que eliminou throttling do HAProxy.

**Aguardando rebuild de índice para AoS (f99743d) — commit não afeta CPU redistrib.**

**Após rebuild:** `make bench-monitor` para confirmar, depois submission com novo docker-compose.

**Current local: 6000 (MAX)** (p99=0.311ms, FP=0, FN=0, ERR=0 — stress test 5000 req/s)
**Last remote: +4353** (p99=44.27ms, FP=0, FN=0 — commit 079bed0)
**Pending remote:** CPU redistrib (lb=0.20, api=0.40) + AVX2 + AoS

---

## Architecture

```
Client → HAProxy :9999 (TCP round-robin)
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

**Resource budget** (hard limits: 1 CPU, 350 MB total):
- Each API instance: 168 MB RAM, **0.40 CPU** (era 0.45)
- HAProxy: 14 MB RAM, **0.20 CPU** (era 0.10) ← chave do ganho

### Fraud pipeline per request

```
JSON decode → fastPath? → TagFromRequest → IvfIndex.Search → pre-rendered response
```

4-bit tag partitions (16 slots, 12 populated): `card_present<<3 | is_online<<2 | unknown_merchant<<1 | has_last_tx`.

Server: raw epoll, GOMAXPROCS(1), GC off (`SetGCPercent(-1)`), `SetMemoryLimit(160MiB)`, Mlockall, SCHED_FIFO.

---

## Index

- **Algorithm:** Flat IVF — adaptive K=n/300 clamped [64, 2048], Lloyd's 20 iters
- **Format:** RNH5-IDX binary (AoS flatVec layout, bbox per cluster) — **rebuild required after f99743d**
- **12 partitions** built by Go `cmd/build_index` from `resources/references.json.gz`
- **NProbeInitial=12**, repair on count ∈ [1,4] → sweep all clusters
- **bboxMin/bboxMax** lower-bound pruning for early cluster skip (SIMD via bpsoaMin/bpsoaMax)
- **sort-within-cluster**: vectors nearest centroid first → tighter worstKey early

### Partition sizes (3M refs total)

| tag | refs | K |
|-----|------|---|
| 0 | 27,887 | 92 |
| 1 | 110,804 | 369 |
| 2 | 4,314 | 64 |
| 3 | 17,213 | 64 |
| 4 | 121,436 | 404 |
| 5 | 489,518 | 1631 |
| 6 | 156,535 | 521 |
| 7 | 627,331 | 2048 |
| 8 | 250,253 | 834 |
| 9 | 1,000,742 | 2048 |
| 10 | 38,532 | 128 |
| 11 | 155,435 | 518 |

---

## Bottleneck analysis

### HAProxy throttling — root cause identificado e resolvido

Monitoring com `docker stats` + HAProxy stats CSV durante stress test revelou:

| Service | Peak CPU% | Limit% | % of Limit |
|---------|-----------|--------|-----------|
| lb | 10.30% | 10% (0.10 core) | **103% → THROTTLED** |
| api1 | 9.28% | 45% (0.45 core) | 21% — OK |
| api2 | 9.42% | 45% (0.45 core) | 21% — OK |

HAProxy com 0.10 CPU + nbthread=1 saturava a ~3000 req/s. APIs tinham 80% de budget sobrando. Redistribuição para lb=0.20/api=0.40 eliminou throttling:

| Antes | Depois |
|-------|--------|
| p99=26.7ms | p99=0.311ms |
| score=4573 | **score=6000 (MAX)** |

CFS throttle do lb causava pausas de ~100ms quando o budget de 0.10 CPU se esgotava. Mesmo sem queue no HAProxy (qcur=0), o throttle causava a cauda longa.

### Otimizações aplicadas (em ordem)

| Commit | Mudança | Impacto medido |
|--------|---------|----------------|
| f20a433 | flat IVF + epoll + GC off | local 0.43ms, remote 38ms |
| 30f0c53 | AVX2 `computeClusterBatch8` (13×) + `distL2i16q` no scan | local 0.35ms |
| f99743d | AoS flatVec — elimina gather loop no scan | aguarda rebuild |
| 80a5fa2 | HAProxy stats endpoint (port 8404, local only) | monitoring infra |
| pending | CPU redistrib: lb=0.20, api=0.40 | **p99=0.311ms, score=6000** |

---

## Score history

| Date | Score | p99 local | p99 remote | FP | FN | Config |
|------|-------|-----------|------------|----|----|--------|
| 2026-05-30 | -6000 | — | 2002ms | 0 | 0 | brute-force KNN |
| 2026-05-30 | +5536 | 0.65ms | — | 19 | 5 | IVF C=4000 nprobe=15 |
| 2026-05-31 | +4051 | — | 30.57ms | 19 | 5 | remote: net/http + GC on |
| 2026-06-04 | +5687 | 0.679ms | — | 10 | 0 | cuML 8000c flat IVF nprobe=40 |
| 2026-06-04 | +5729 | 0.394ms | — | 4 | 1 | IVF_H2 + bbox pruning |
| 2026-06-04 | +5909 | 0.606ms | — | 1 | 0 | 4-way split by unknown_merchant |
| 2026-06-05 | +4084 | — | 59ms | 2 | 0 | remote: old IVFH (commit 491f131) |
| 2026-06-05 | +4411 | 0.43ms | 38ms | 0 | 0 | flat IVF + epoll + GC off (commit f20a433) |
| 2026-06-05 | 6000 | 0.43ms | — | 0 | 0 | local max (partition pre-partitioning) |
| 2026-06-05 | — | 0.35ms | pending | 0 | 0 | AVX2 bbox batch + distL2i16q (commit 30f0c53) |
| 2026-06-05 | +4353 | — | 44.27ms | 0 | 0 | commit 079bed0 (pré CPU redistrib) |
| 2026-06-05 | **6000** | **0.311ms** | pending | 0 | 0 | **CPU redistrib: lb=0.20, api=0.40** |

---

## Fixed constraints (spec — never change)

- `k=5`: test labels generated with exact brute-force k=5.
- `threshold=0.6`: fixed in `DETECTION_RULES.md`.
- `fraud_score = fraudCount / 5.0`: always divides by 5.
