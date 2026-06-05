# PROGRESS

**Last updated:** 2026-06-05 (post-AoS refactor)

## Status

Detection perfeita (FP:0, FN:0) local e remoto. Remote P99=38ms dominado por CFS throttling: servidor consome budget de 0.45 CPU antes do fim do período, causando espera de até ~100ms. Confirmado localmente com `test/stress.js` — p99=42ms sob carga extrema (3000+ req/s).

**Duas otimizações recentes em main, aguardando rebuild de índice:**
1. `computeClusterBatch8` AVX2 (commit 30f0c53) — cluster bbox 13× mais rápido
2. AoS flatVec (commit f99743d) — elimina gather loop no scan de vetores

**Após rebuild:** `make bench-stress` para medir localmente, depois submission.

**Current local: 6000 (MAX)** (p99=0.35ms, FP=0, FN=0, ERR=0 — pré-AoS)
**Last remote: +4411** (p99=38.75ms, FP=0, FN=0 — commit f20a433)
**Pending remote: commit 30f0c53** (AVX2; issue #9111)

---

## Architecture

```
Client → HAProxy :9999 (TCP round-robin)
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

Each instance: 168 MB RAM, 0.45 CPU. HAProxy: 14 MB, 0.10 CPU. Total: 1 CPU, 350 MB.

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

## Bottleneck analysis (confirmado via stress test)

Root cause do remote P99=38ms: **CFS throttling**. Stress test local (`test/stress.js`, 3000→5000 req/s) replicou exatamente o cenário: p99=42ms com p50=0.16ms. A maioria das requisições é rápida; a cauda longa vem de lotes que esgotam o budget de 0.45 CPU antes do próximo período CFS (~100ms).

### Otimizações aplicadas (em ordem)

| Commit | Mudança | Impacto medido |
|--------|---------|----------------|
| f20a433 | flat IVF + epoll + GC off | local 0.43ms, remote 38ms |
| 30f0c53 | AVX2 `computeClusterBatch8` (13×) + `distL2i16q` no scan | local 0.35ms |
| f99743d | AoS flatVec — elimina gather loop no scan | aguarda rebuild |

### computeClusterPacked (resolvido)

`computeClusterBatch8`: 8 clusters/iter via SIMD em `bpsoaMin/bpsoaMax`. K=2048 → 256 grupos × 5ns = 1.3μs vs ~17μs escalar. **~13× mais rápido.**

### scanClusterGather — gather eliminado (f99743d)

**Antes (SoA):** 7 leituras espalhadas por vetor (`pairs[p][2*vi]`). Para 12 probes × 488 vetores: ~41K acessos a 7 arrays distintos de 4MB → cache thrashing.

**Depois (AoS):** `distL2i16q(ix.flatVec, vi*16, q)` direto. Vetor contíguo (32 bytes), scan sequencial dentro do cluster → L1-warm. Estimativa: 4-8× speedup no scan.

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
| 2026-06-05 | +4411 | 0.43ms | 38ms | 0 | 0 | **flat IVF + epoll + GC off (commit f20a433)** |
| 2026-06-05 | 6000 | 0.43ms | — | 0 | 0 | local max (partition pre-partitioning) |
| 2026-06-05 | — | 0.35ms | pending | 0 | 0 | AVX2 bbox batch + distL2i16q (commit 30f0c53) |

---

## Fixed constraints (spec — never change)

- `k=5`: test labels generated with exact brute-force k=5.
- `threshold=0.6`: fixed in `DETECTION_RULES.md`.
- `fraud_score = fraudCount / 5.0`: always divides by 5.
