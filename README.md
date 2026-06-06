# gopher-fraud-detection

Fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).
Hard limits: 1 CPU, 350 MB RAM total.

---

## Index

3M labeled transactions split into 12 partitions via 4-bit tag:
`card_present | is_online | unknown_merchant | has_last_tx`.

Each partition has its own flat IVF index — adaptive K (n/300, clamped [64, 2048]),
Lloyd's 20 iterations, int16 vectors (×10000 scale), AoS layout with per-cluster bounding boxes.

## Search

Tagged at request time → routed to partition → approximate KNN (k=5, L2, nprobe=12).

- **Bbox pruning:** clusters whose lower-bound distance exceeds the current worst neighbor are skipped.
- **SIMD:** two hand-written AVX2 routines in `internal/search/knn_amd64.s`. `distL2i16q` computes
  exact L2² between two int16 vectors using `VPSUBW + VPMADDWD` — no float conversion ever.
  `computeClusterBatch8` scores 8 cluster bboxes in parallel via `VPBROADCASTD`, feeding the
  pruning step. Together ~13× faster than scalar Go.
- **Repair pass:** if result count is ambiguous (1–4), all clusters are swept.
- `fraud_score = fraudCount / 5.0`, approved if < 0.6 (fixed by spec).

## Why a Go load balancer

HAProxy was the actual bottleneck — not the search. At ~3000 req/s it saturated its 0.10 CPU
budget, got CFS-throttled, and paused requests up to 100ms. The Go LB uses raw epoll +
`SO_REUSEPORT` + `SCHED_FIFO`, burning ~2% CPU at peak. That freed one core to redistribute:
`lb=0.20 / api1=0.40 / api2=0.40`. Result: p99 dropped from 26ms → 0.3ms, score 4500 → 6000.

## Curiosities

- **GC off.** `SetGCPercent(-1)` + `SetMemoryLimit(160 MiB)`. Index allocated once, never freed.
- **Real-time scheduling.** `SCHED_FIFO` via `CAP_SYS_NICE` — epoll workers never preempted mid-request.
- **No framework.** Raw epoll loop over unix sockets. No Gin, no Fiber, no Zap.
- **Score 6000/6000 locally.** FP=0, FN=0, p99=0.3ms. Bottleneck is the remote test harness.
