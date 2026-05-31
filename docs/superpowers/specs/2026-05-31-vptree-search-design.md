# VP-Tree Exact Search — Design Spec

**Date:** 2026-05-31
**Status:** Approved
**Goal:** Replace IVF approximate search with VP-tree exact search to eliminate FP/FN errors and reduce remote p99 latency from 30ms to <1ms.

---

## Problem

Current IVF index (C=4000, nprobe=15) achieves FP:19, FN:5 locally and p99=30ms on remote (vs 0.65ms local). Root cause: IVF is approximate — queries near cluster boundaries miss true top-5 neighbors. Remote CPU is also weaker, causing CFS throttling under load. Both problems solved by exact search with fewer comparisons.

---

## Solution: VP-Tree (Vantage Point Tree)

VP-tree partitions space by **actual distance from a pivot**, not axis-aligned planes. Every node picks a pivot, computes the median distance tau to all points in the partition, and splits into inner ball (dist ≤ tau) and outer ball (dist > tau). Queries use branch-and-bound pruning via the triangle inequality — exact, not approximate.

**Why exact:** every pruning decision is: `minPossibleDist(subtree, query) >= currentKthBestDist`. The triangle inequality gives a tight lower bound. No approximation involved. FP:0, FN:0 is the theoretical guarantee.

**Comparison count:** with 3M vectors at depth 22 and leaf=16, once the k=5 heap fills (after ~10 comparisons), 80-95% of subtrees are pruned. Expected ~50-200 comparisons/query vs ~15,250 for IVF.

---

## Architecture

No new packages. VP-tree lives alongside IVF in `internal/search/`. Both implement the same `KNN([14]float32, int) int` signature. `cmd/api/main.go` auto-detects format by magic bytes at startup.

```
internal/search/
  index.go       — existing: IVFIndex + LoadIVFIndex
  knn.go         — existing: IVFIndex.KNN
  vp_index.go    — new: VPNode, VPIndex, LoadVPIndex
  vp_knn.go      — new: VPIndex.KNN (iterative DFS, branch-and-bound)
  vp_knn_test.go — new: correctness vs brute force
  vp_index_test.go — new: loader round-trip

ml/
  build_index.py — modified: --algo {vptree,ivf} flag, vptree default
```

`cmd/api/main.go` startup logic:

```go
magic, _ := readMagic(indexPath)  // peek first 4 bytes
switch magic {
case "IVF1":
    idx, err = search.LoadIVFIndex(indexPath)
case "VPT1":
    idx, err = search.LoadVPIndex(indexPath)
default:
    log.Fatal("unknown index format")
}
```

Both index types satisfy an interface defined in `internal/search/index.go` (alongside existing IVF code):

```go
// Index is the common interface for both IVFIndex and VPIndex.
type Index interface {
    KNN(query [14]float32, k int) int
}
```

`readMagic` is a small helper in `cmd/api/main.go`:

```go
func readMagic(path string) (string, error) {
    f, err := os.Open(path)
    if err != nil {
        return "", err
    }
    defer f.Close()
    var buf [4]byte
    _, err = io.ReadFull(f, buf[:])
    return string(buf[:]), err
}
```

---

## Binary Format: VPT1

All values little-endian.

```
[4B]  "VPT1" magic
[4B]  uint32 N          — total vectors
[4B]  uint32 nodeCount  — number of nodes (~375K for N=3M, leaf=16)
[4B]  uint32 leafSize   — leaf threshold (16)

[nodeCount × 40B] node array (DFS order):
  [4B]  float32 tau       — split radius (actual Euclidean distance, not squared)
                            for leaves: unused (zero)
  [4B]  uint32  childOff  — internal: right child node index
                            leaf: start index in vectors array
  [2B]  uint16  count     — 0 = internal node; >0 = leaf (number of vecs to scan, ≤16)
  [2B]  _pad
  [28B] int16[14] vec     — pivot vector scaled ×10000 (internal nodes only; zero for leaves)

[N × 28B]  int16[14] vectors  — DFS-reordered, scaled ×10000
[N × 1B]   uint8     labels   — DFS-reordered (0=legit, 1=fraud)
```

**Go struct (in `vp_index.go`) — maps directly onto binary layout:**

```go
type VPNode struct {
    Tau      float32    // actual split distance (not squared)
    ChildOff uint32     // right child node idx (internal) | vec array start (leaf)
    Count    uint16     // 0 = internal; >0 = leaf (# vecs ≤ 16)
    _pad     [2]byte
    Vec      [14]int16  // pivot vector ×10000; zeroed for leaf nodes
}

type VPIndex struct {
    Nodes    []VPNode
    Vectors  []int16  // DFS-reordered, int16 ×10000
    Labels   []uint8  // DFS-reordered
}
```

**Node addressing:**
- Left child of node at index `i`: always `i+1` (implicit, no storage needed)
- Right child of node at index `i`: `node[i].childOff`
- Leaf detection: `node[i].count > 0`

**Memory layout per instance:**
```
Node array:   ~375K × 40B  =  15MB  ← fits entirely in L3 cache (32MB)
Vector array: 3M × 28B     =  84MB
Label array:  3M × 1B      =   3MB
Total:                       102MB  < 168MB budget ✓
```

---

## Python Build Pipeline

`ml/build_index.py` gains `--algo` flag:

```bash
uv run ml/build_index.py            # VP-tree (default)
uv run ml/build_index.py --algo ivf # IVF (unchanged)
```

**Phase 1 — Recursive tree construction:**

```python
LEAF_SIZE = 16

def build(indices):
    if len(indices) <= LEAF_SIZE:
        return ('leaf', indices)

    pivot_pos = rng.integers(len(indices))
    pivot_idx = indices[pivot_pos]
    pivot_vec = vectors[pivot_idx]  # float32

    diffs = vectors[indices] - pivot_vec
    dists_sq = np.einsum('ij,ij->i', diffs, diffs)
    dists = np.sqrt(dists_sq)
    tau = float(np.median(dists))

    mask = dists <= tau
    if mask.all() or (~mask).all():      # degenerate: split at midpoint
        mid = len(indices) // 2
        inner, outer = indices[:mid], indices[mid:]
    else:
        inner, outer = indices[mask], indices[~mask]

    return ('node', pivot_idx, tau, build(inner), build(outer))
```

Pivot is included in the left (inner) partition via natural `dist(pivot, pivot) = 0 ≤ tau`.
Recursion depth ≤ log₂(3M/16) ≈ 18 — well within Python default limit.

**Phase 2 — DFS serialization into flat arrays:**

```python
nodes = []       # list of dicts → packed to binary
vec_order = []   # global indices in DFS traversal order

def serialize(tree):
    if tree[0] == 'leaf':
        _, indices = tree
        ni = len(nodes)
        vec_start = len(vec_order)
        vec_order.extend(indices.tolist())
        nodes.append({'leaf': True, 'childOff': vec_start,
                      'count': len(indices), 'tau': 0.0, 'vec': np.zeros(14, dtype=np.int16)})
        return ni

    _, pivot_idx, tau, left, right = tree
    ni = len(nodes)
    nodes.append(None)          # placeholder
    serialize(left)             # left child at ni+1 (implicit)
    right_ni = len(nodes)
    vec_int16 = np.clip(np.round(vectors[pivot_idx] * 10000), -32768, 32767).astype(np.int16)
    nodes[ni] = {'leaf': False, 'tau': tau, 'childOff': right_ni,
                 'count': 0, 'vec': vec_int16}
    serialize(right)
    return ni
```

**Phase 3 — Write VPT1:**

Reorder `vectors_int16` and `labels` by `vec_order`, then pack header + node array + vectors + labels.

**Estimated build time:** 5-15 min. Bottleneck: Python recursion overhead for ~375K calls.
**Peak memory on build machine:** ~300MB (float32 vectors + Python tree objects + output arrays).

---

## Go Query: VPIndex.KNN

**Iterative DFS, explicit stack, zero allocations per request.**

```go
const vpInvScale = float32(1.0 / 10000.0)

type vpStackEntry struct {
    nodeIdx uint32
    minDist float32  // min possible actual distance from query to any point in subtree
}

func (idx *VPIndex) KNN(query [14]float32, k int) int {
    var stackArr [40]vpStackEntry   // depth ≤ 18 × 2 branches + headroom; stack-allocated
    stack := stackArr[:0]

    var topArr [5]knnEntry          // k=5, stack-allocated
    top := topArr[:0]
    maxDistSq := float32(math.MaxFloat32)
    maxRadius  := float32(math.MaxFloat32)  // sqrt(maxDistSq); updated on each heap change
    maxPos := 0

    // extract query to locals — avoids repeated bounds checks
    q0, q1, ..., q13 := query[0], query[1], ..., query[13]

    stack = append(stack, vpStackEntry{0, 0})

    for len(stack) > 0 {
        e := stack[len(stack)-1]
        stack = stack[:len(stack)-1]

        if len(top) == k && e.minDist >= maxRadius {
            continue  // entire subtree pruned by triangle inequality
        }

        node := idx.Nodes[e.nodeIdx]

        if node.Count > 0 {
            // LEAF: unrolled 14-dim linear scan with early exit at dim 7
            base := int(node.ChildOff) * dims
            for vi := int(node.ChildOff); vi < int(node.ChildOff)+int(node.Count); vi++ {
                _ = idx.Vectors[base+13]  // bounds-check hint
                d0 := q0 - float32(idx.Vectors[base])*vpInvScale
                distSq := d0 * d0
                // ... (same unrolled pattern as IVF knn.go, early exit at dim 7)
                // update heap; if heap changes, update maxRadius = sqrt(maxDistSq)
                base += dims
            }
            continue
        }

        // INTERNAL: compute actual distance to pivot (1 sqrt per internal node visited)
        distSq := unrolledPivotDistSq(q0..q13, node.Vec)
        dist   := float32(math.Sqrt(float64(distSq)))
        tau    := node.Tau

        leftChild  := e.nodeIdx + 1    // always adjacent (implicit)
        rightChild := node.ChildOff

        if dist <= tau {
            // query inside ball: inner (left) is closer
            outerMin := tau - dist
            if outerMin < maxRadius {
                stack = append(stack, vpStackEntry{rightChild, outerMin})  // farther, popped later
            }
            stack = append(stack, vpStackEntry{leftChild, 0})              // closer, popped first
        } else {
            // query outside ball: outer (right) is closer
            innerMin := dist - tau
            if innerMin < maxRadius {
                stack = append(stack, vpStackEntry{leftChild, innerMin})   // farther, popped later
            }
            stack = append(stack, vpStackEntry{rightChild, 0})             // closer, popped first
        }
    }

    fraudCount := 0
    for _, e := range top {
        if e.label == 1 {
            fraudCount++
        }
    }
    return fraudCount
}
```

**Optimizations carried forward from IVF knn.go:**
- `vpInvScale` multiply instead of `/ 10000.0`
- Unrolled 14-dim distance loops
- Bounds-check hint `_ = vecs[base+13]`
- Early exit at dim 7 during leaf scan
- Query locals `q0..q13`

**New for VP-tree:**
- Stack-allocated `[40]vpStackEntry` (replaces heap-allocated slice)
- `maxRadius` (actual distance) maintained alongside `maxDistSq` (squared); updated only on heap change
- 1 `math.Sqrt` per internal node visited (~50-200 total, ≈4μs overhead)
- "Push farther first" ordering ensures closer subtree is always explored first

---

## Testing

### `vp_knn_test.go`

```
TestVPKNNMatchesBruteForce
  Build synthetic VPIndex (N=500, ~30% fraud labels, random float32 vecs)
  Run 200 random queries
  For each: compare VP-tree fraudCount against brute-force fraudCount
  Must match exactly — any mismatch = correctness bug

TestVPKNNLeafOnly
  N=10 (single leaf, no tree traversal)
  Verify fraudCount matches brute force

TestVPKNNExactPivotHit
  Query vector identical to a known pivot
  Verify it appears in top-5 (dist=0 always wins)

TestVPKNNDegenerateSplit
  All vectors equidistant from pivot (triggers midpoint split)
  Verify brute-force match still holds

TestVPKNNAllFraud / TestVPKNNAllLegit
  fraudCount must be 5 / 0 respectively
```

**Acceptance bar:** `TestVPKNNMatchesBruteForce` must pass all 200 queries before merge.

### `vp_index_test.go`

```
TestLoadVPIndex        — write known VPT1 bytes, load, verify all fields
TestLoadVPIndexBadMagic — wrong magic → descriptive error
TestLoadVPIndexSizeMismatch — truncated file → descriptive error
```

---

## Score Projection

| Environment | Current (IVF) | Target (VP-tree) |
|---|---|---|
| FP / FN | 19 / 5 | 0 / 0 |
| det_score | 2536 | 3000 |
| local p99 | 0.65ms | <0.5ms est. |
| remote p99 | 30ms | <1ms est. |
| remote score | 4051 | **5700-6000** |

---

## Constraints (unchanged)

- k=5, threshold=0.6, fraud_score = fraudCount / 5.0 — fixed by spec, never change
- Index path: `index/references.bin` — same filename, different magic bytes
- Per-instance RAM budget: 168MB — VP-tree uses 102MB ✓
- Go stdlib only — no external search libraries
