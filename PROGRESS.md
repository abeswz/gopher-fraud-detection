# PROGRESS

**Last updated:** 2026-05-31

## Status

VP-tree implementation complete. All 21 tests pass.
Awaiting index rebuild (`uv run ml/build_index.py`) and remote submission.

Previous remote: **+4051 / +6000** (p99=30ms, FP:19, FN:5)
Target with VP-tree: **~5700-6000** (p99<1ms, FP:0, FN:0 exact)

---

## Architecture

```
Client → HAProxy :9999 (TCP round-robin)
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

Each instance: 168 MB RAM, 0.45 CPU. HAProxy: 14 MB, 0.10 CPU. Total: 1 CPU, 350 MB.

---

## What is implemented

- `cmd/api/main.go` — magic-byte auto-detect (IVF1/VPT1), loads correct index, serves unix socket
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim feature vector
- `internal/search/index.go` — `IVFIndex` + `LoadIVFIndex` (IVF1 format) + `Index` interface
- `internal/search/knn.go` — IVF KNN (nprobe=15, C=4000), fully optimized
- `internal/search/vp_index.go` — `VPNode`, `VPIndex`, `LoadVPIndex` (VPT1 binary format)
- `internal/search/vp_knn.go` — `VPIndex.KNN` iterative DFS, branch-and-bound, zero heap allocs
- `internal/service/fraud_detection.go` — uses `search.Index` interface, k=5, threshold=0.6 (FIXED BY SPEC)
- `internal/handler/fraud_score.go` — pre-computed 6-entry response array, zero JSON encoding
- `ml/build_index.py` — `--algo {vptree,ivf}` flag (vptree default); IVF: MiniBatchKMeans(C=4000, n_init=3); VP: recursive DFS tree → VPT1 binary
- `Makefile` — `index`, `bench`, `bench-fast`, `submission` targets
- `references/tools/profile.sh` — CPU/mem profile (`vp|ivf` × `serial|parallel`); output → `references/performance/`
- `references/tools/trace.sh` — execution trace (`vp|ivf`); output → `references/performance/`
- Tests — 21 unit tests

---

## Fixed constraints (spec — never change)

- `k=5`: test labels generated with exact brute-force k=5. Changing k diverges from ground truth → more FP/FN.
- `threshold=0.6`: explicitly fixed in `DETECTION_RULES.md`.
- `fraud_score = fraudCount / 5.0`: always divides by 5.

---

## Score history

| Date | Score | p99 | p99_score | FP | FN | Config |
|---|---|---|---|---|---|---|
| 2026-05-30 | -6000 | 2002ms | -3000 | 0 | 0 | brute-force KNN |
| 2026-05-30 | +5322 | 1.64ms | 2786 | 19 | 5 | IVF C=4000 nprobe=15 |
| 2026-05-30 | +5070 | 3.13ms | 2504 | 15 | 4 | nprobe=20 (CPU saturated) |
| 2026-05-30 | +5292 | 1.76ms | 2755 | 19 | 5 | nprobe=15 + pre-computed response |
| **2026-05-30** | **+5536** | **0.65ms** | **3000 (MAX)** | 19 | 5 | **nprobe=15 + KNN optimized** |
| 2026-05-31 | +4051 | 30.57ms | 1514 | 19 | 5 | **remote submission** (same code) |
| 2026-05-31 | +3234 | 582ms | 234 | 0 | 0 | **VP-tree LEAF_SIZE=16 local** — detection perfeita, latência catastrófica |
| 2026-05-31 | +3188 | 648ms | 188 | 0 | 0 | **VP-tree LEAF_SIZE=256 local** — pior que LEAF_SIZE=16, hipótese: cache thrashing |

---

## Remote vs local gap analysis

Remote p99=30ms vs local p99=0.96ms (31× slower). Detection identical → problem is pure compute speed.

Root cause: remote CPU is weaker. IVF Phase 1 scans 4000 centroids (224KB), Phase 2 scans ~11,250 vectors per query. Under 0.45 CPU with concurrent load, CFS bandwidth throttling causes p99 spikes.

VP-tree eliminates this: ~50-200 vector comparisons per query vs ~15,250 for IVF. Branch-and-bound pruning via triangle inequality. Exact results → FP:0, FN:0.

Scoring formula (from EVALUATION.md):
```
score_p99 = 1000 × log₁₀(1000 / max(p99_ms, 1))
```

| remote p99 target | p99_score | total score |
|---|---|---|
| 30ms (current IVF) | 1514 | 4051 |
| 3ms | 2523 | 5523 (det_score=3000) |
| 1ms | 3000 | 6000 (MAX) |

---

## VP-tree binary format (VPT1)

```
[4B]   "VPT1" magic
[4B]   uint32 N          — total vectors
[4B]   uint32 nodeCount
[4B]   uint32 leafSize   — stored, not used at query time

[nodeCount × 40B] node array:
  [4B]  float32 tau       — split radius; 0 for leaves
  [4B]  uint32  childOff  — right child node index (internal) | vec array start (leaf)
  [2B]  uint16  count     — 0 = internal; >0 = leaf (# vecs ≤ 16)
  [2B]  pad
  [28B] int16[14] vec     — pivot ×10000; zeroed for leaves

[N × 28B]  int16[14] vectors — DFS-reordered, ×10000
[N × 1B]   uint8     labels  — DFS-reordered
```

Memory: ~135 MB per instance (84 MB vectors + 48 MB nodes + 3 MB labels) < 168 MB budget.

---

## VP-tree performance analysis (dados medidos)

| Config | µs/op (serial) | p99 k6 local | Causa |
|--------|---------------|-------------|-------|
| LEAF_SIZE=16 | 178µs | 582ms | 524K nós, 21MB node array, muitos sqrt() |
| LEAF_SIZE=256 | ? | 648ms | Mais vectors por folha, sem ganho |

**Conclusão:** VP-tree não escala sob carga concorrente neste workload.

Micro-bench serial (178µs) vs p99 real (582ms) → gap de 3.200x. Causa: **cache thrashing concorrente**.

- 84MB de vetores (DFS-reordenados) = acesso aleatório sob múltiplas goroutines simultâneas
- Cada query toca partes espalhadas dos 84MB → cache L3 thrash com 16+ goroutines concorrentes
- IVF: centroids 224KB (quente em L1), cluster data sequencial → cache-friendly sob concorrência

Aumentar LEAF_SIZE não resolve: o gargalo é o array de vetores (84MB), não os nós.

## Análise IVF vs VP-tree sob concorrência

IVF funciona bem concorrentemente porque:
- Fase 1 (centroids): 224KB sempre em cache, leitura compartilhada entre goroutines
- Fase 2 (cluster scan): cada goroutine lê cluster específico → localidade espacial

VP-tree falha concorrentemente porque:
- DFS path de cada query = sequência única de posições nos 84MB
- Múltiplas goroutines = múltiplas sequências aleatórias → cache miss constante

---

## Next steps

**Caminho recomendado:** voltar ao IVF e investigar o gap remoto (30ms).

Opções para melhorar score remoto com IVF:
1. `GOMAXPROCS=1` no Dockerfile — evita CFS thrashing com 0.45 CPU (testado local: p99=1.37ms, pode ajudar remoto)
2. Investigar por que remoto é 31× mais lento que local com mesmo código

---

## KNN performance history

| Config | µs/op serial | p99 k6 local | Notes |
|--------|-------------|-------------|-------|
| Brute-force | ? | 2002ms | original |
| IVF C=4000 nprobe=15 unopt | 79µs | 1.64ms | baseline |
| IVF C=4000 nprobe=20 | 88µs | 3.13ms | CPU saturado |
| IVF nprobe=15 + pre-comp resp | ? | 1.76ms | |
| **IVF nprobe=15 + KNN otimizado** | **~40µs** | **0.65ms** | **melhor local** |
| VP-tree LEAF_SIZE=16 | 178µs | 582ms | cache thrash |
| VP-tree LEAF_SIZE=256 | ? | 648ms | pior ainda |
