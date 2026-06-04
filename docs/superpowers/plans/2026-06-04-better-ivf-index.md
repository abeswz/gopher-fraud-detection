# Better IVF Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MiniBatchKMeans with standard KMeans (8000 clusters, n_init=10) to produce a higher-quality IVF index, then tune nprobe down from 20 to 15 for lower p99 with equivalent or better recall.

**Architecture:** Offline index rebuild only — the Go binary and HTTP serving code are unchanged except for the `nprobe` constant in `knn.go`. The new index uses the same IVF2 binary format; `LoadIVFIndex` reads cluster count dynamically from the header so no Go changes are needed at load time.

**Tech Stack:** Python 3 (uv), scikit-learn KMeans, numpy, Go 1.26

---

## File Map

| File | Change |
|------|--------|
| `ml/build_index.py` | Replace `MiniBatchKMeans` → `KMeans`, `N_CLUSTERS=8000`, add `N_INIT=10` |
| `internal/search/knn.go` | `nprobe` constant: `20` → `15` (after benchmark confirms it) |
| `references/bench/result.csv` | Append new nprobe sweep rows after build |

---

## Task 1: Update `ml/build_index.py`

**Files:**
- Modify: `ml/build_index.py`

- [ ] **Step 1: Replace constants and import**

Replace the top of `ml/build_index.py` (lines 1–9) so it reads:

```python
import gzip
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

N_CLUSTERS = 8000
N_INIT     = 10
NPROBE_DEFAULT = 15
```

- [ ] **Step 2: Replace the KMeans call in `build_ivf`**

Replace the MiniBatchKMeans block (lines 20–28):

```python
    print(f"Fitting KMeans with {N_CLUSTERS} clusters (n_init={N_INIT})...")
    km = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=N_INIT,
        max_iter=300,
        verbose=1,
    )
    assignments = km.fit_predict(vectors)
    centroids = km.cluster_centers_.astype(np.float32)  # shape: (C, 16)
    print(f"KMeans done. Centroids: {centroids.shape}")
```

- [ ] **Step 3: Verify the final file looks correct**

Run a syntax check (no index build yet):

```bash
uv run --project ml python -c "import ml.build_index; print('OK')" 2>/dev/null || \
  uv run --project ml python -c "
import ast, pathlib
src = pathlib.Path('ml/build_index.py').read_text()
ast.parse(src)
print('Syntax OK')
"
```

Expected output: `Syntax OK`

- [ ] **Step 4: Commit the script change**

```bash
git add ml/build_index.py
git commit -m "perf(ml): replace MiniBatchKMeans with KMeans, 8000 clusters, n_init=10"
```

---

## Task 2: Build the New Index

**Files:**
- Generated: `index/references.bin` (gitignored)

This step takes **3–7 hours** on CPU (16 cores). Run it and let it complete. If you have cuML/RAPIDS installed you can swap the import for `from cuml.cluster import KMeans` (same API) and finish in ~20 min — but that is optional.

- [ ] **Step 1: Start the index build**

```bash
make index
```

Expected output (will be verbose because of `verbose=1`):
```
Loading records...
Loaded 3000000 records
Fitting KMeans with 8000 clusters (n_init=10)...
Initialization complete
Iteration 0, inertia XXXXXX
Iteration 1, inertia XXXXXX
...
KMeans done. Centroids: (8000, 16)
Writing IVF2 index...
3000000 vectors, 8000 clusters → index/references.bin (XXX.X MB)
Avg cluster size: 375, nprobe=15 → ~5625 vecs/query
```

`make index` wraps `uv run --project ml ml/build_index.py`.

- [ ] **Step 2: Verify index file was written**

```bash
ls -lh index/references.bin
```

Expected: file exists, size roughly same as before (dominant cost is the N×16 int16 vectors — ~87 MB for 3M vecs). The centroid block grows from 4000×16×4 = 256 KB to 8000×16×4 = 512 KB — negligible difference.

- [ ] **Step 3: Verify the Go binary loads the new index without errors**

```bash
go test ./internal/search/ -run TestLoad -v 2>&1 | head -20
```

If no `TestLoad` exists, run all search tests:

```bash
go test ./internal/search/ -v -count=1 2>&1 | tail -20
```

Expected: all tests PASS. The loader reads `C` from the file header dynamically (`struct.pack("<II", N_CLUSTERS, n)` in the writer) so 8000 clusters load transparently.

---

## Task 3: Benchmark nprobe Sweep

**Files:**
- Read: `references/bench/result.csv` (append new rows)

Run the local bench for nprobe ∈ {10, 12, 15, 20} and record results. The bench uses `make bench-fast` (skips the index build step, uses the already-running Docker stack).

- [ ] **Step 1: Understand the bench workflow**

`make bench-fast`:
1. Tears down and rebuilds the Docker stack with the new index
2. Waits for `/ready`
3. Runs `k6 run test/test.js`
4. Prints `p99 / score / FP / FN / ERR` summary

The nprobe constant lives in `internal/search/knn.go:4`. Each nprobe value requires recompiling and rebuilding the Docker image.

- [ ] **Step 2: Bench nprobe=20 (baseline on new index)**

Leave `knn.go` at `nprobe = 20`. Run:

```bash
make bench-fast
```

Record the output line. Example format:
```
p99:0.45 score:XXXX FP:XX FN:X ERR:0
```

Append a row to `references/bench/result.csv` — copy the full JSON fields from `test/results.json` if the collect script isn't wired up, or run:

```bash
jq -r '[20, .expected.total, .expected.fraud_count, .expected.legit_count,
  .expected.fraud_rate, .expected.legit_rate, .expected.edge_case_count, .expected.edge_case_rate,
  .p99, .scoring.breakdown.false_positive_detections, .scoring.breakdown.false_negative_detections,
  .scoring.breakdown.true_positive_detections, .scoring.breakdown.true_negative_detections,
  .scoring.breakdown.http_errors, .scoring.failure_rate, .scoring.breakdown.weighted_errors_E,
  .scoring.error_rate_epsilon, .scoring.p99_score.value, .scoring.p99_score.cut_triggered,
  .scoring.detection_score.value, .scoring.detection_score.cut_triggered,
  .scoring.detection_score.rate_component, .scoring.detection_score.absolute_penalty,
  .scoring.final_score] | @csv' test/results.json >> references/bench/result.csv
```

**Acceptance gate:** `final_score` at nprobe=20 ≥ 5581 (old baseline). If worse, KMeans quality insufficient — see Task 3 Step 7 fallback.

- [ ] **Step 3: Bench nprobe=15**

Edit `internal/search/knn.go` line 4:

```go
const (
	nprobe   = 15
	invScale = float32(1.0 / 10000.0)
)
```

Run:

```bash
make bench-fast
```

Append result to CSV (same jq command, change `20` to `15`).

**Target:** `final_score` ≥ 5550 AND `p99` < p99 at nprobe=20.

- [ ] **Step 4: Bench nprobe=12**

Edit `knn.go`: `nprobe = 12`. Run bench, append CSV row.

- [ ] **Step 5: Bench nprobe=10**

Edit `knn.go`: `nprobe = 10`. Run bench, append CSV row.

- [ ] **Step 6: Pick the best nprobe**

Decision rule (in order):
1. If nprobe=12 achieves `final_score` ≥ 5550 → use 12
2. Else if nprobe=15 achieves `final_score` ≥ 5550 → use 15
3. Else stay at nprobe=20

- [ ] **Step 7: Fallback (only if nprobe=20 on new index < 5581)**

The new 8000-cluster index at the same nprobe performed worse than the old 4000-cluster index. Options:
- Retry with `N_INIT=20` in `build_index.py` (double build time but better centroid quality)
- Or revert to `N_CLUSTERS=6000` as a compromise

Do not proceed to Task 4 without a passing nprobe=20 baseline on the new index.

---

## Task 4: Set Final nprobe and Commit

**Files:**
- Modify: `internal/search/knn.go:4`

- [ ] **Step 1: Set the chosen nprobe**

Edit `internal/search/knn.go` to the nprobe chosen in Task 3 Step 6. Example for nprobe=15:

```go
const (
	nprobe   = 15
	invScale = float32(1.0 / 10000.0)
)
```

- [ ] **Step 2: Run full test suite**

```bash
go test ./... -count=1
```

Expected: all tests PASS. The nprobe constant only affects which stack-allocated array size is used in `KNN` — the algorithm is unchanged, existing tests remain valid.

- [ ] **Step 3: Build the binary**

```bash
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o fraud-api ./cmd/api
```

Expected: exits 0, no warnings.

- [ ] **Step 4: Run final bench**

```bash
make bench-fast
```

Confirm the score and p99 match the benchmark recorded in Task 3.

- [ ] **Step 5: Commit**

```bash
git add internal/search/knn.go references/bench/result.csv
git commit -m "perf(search): lower nprobe to <CHOSEN> — better IVF index needs fewer probes"
```

Replace `<CHOSEN>` with the actual value (12, 15, or 20).

---

## Task 5: Update PROGRESS.md

**Files:**
- Modify: `PROGRESS.md`

Per `CLAUDE.md`: after each architectural change, rewrite `PROGRESS.md` with the latest state. This is a rewrite (not an append).

- [ ] **Step 1: Rewrite PROGRESS.md**

Replace the full contents of `PROGRESS.md` with current state:
- New index: KMeans 8000 clusters, n_init=10
- nprobe: chosen value
- Latest bench scores (nprobe sweep table)
- Current p99 and final_score

- [ ] **Step 2: Commit**

```bash
git add PROGRESS.md
git commit -m "docs(progress): update after KMeans index rebuild + nprobe tuning"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|-----------------|------------------|
| Replace MiniBatchKMeans → KMeans | Task 1 |
| N_CLUSTERS=8000, N_INIT=10 | Task 1 |
| max_iter=300, verbose=1 | Task 1 |
| Build new index | Task 2 |
| Verify loader works (reads C dynamically) | Task 2 Step 3 |
| Benchmark nprobe ∈ {10, 12, 15, 20} | Task 3 |
| Record results in result.csv | Task 3 |
| Acceptance criteria (score ≥ 5550 at nprobe=15) | Task 3 Step 6 |
| Fallback if new index worse at nprobe=20 | Task 3 Step 7 |
| Update knn.go nprobe constant | Task 4 |
| Update PROGRESS.md | Task 5 |

No gaps found. Placeholders: none. Types/signatures: no new types introduced — only constant and import changes.
