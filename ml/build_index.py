import gzip
import json
import struct
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.cluster import KMeans
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

# 2-level IVF hyperparameters
K2_FIRST = 32   # micro clusters per macro, first_tx sub-indexes
K2_SUBSEQ = 16  # micro clusters per macro, subsequent sub-indexes

# DIM index for static partitioning: unknown_merchant (1=unknown, 0=known)
DIM_UNKNOWN_MERCHANT = 11

DIMS = 16  # padded feature dims (14 features + 2 zero-padding)
SCALE = 10000  # int16 quantization factor


def choose_k1(N: int, K2: int = 16, target_leaf_size: int = 500) -> int:
    """Scale K1 so avg leaf size ≈ target_leaf_size."""
    k1 = max(4, round(N / (target_leaf_size * K2)))
    for nice in [4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512]:
        if nice >= k1:
            return nice
    return 512


def detect_null_tx_mask(vectors: np.ndarray) -> np.ndarray:
    """True where both dim5 and dim6 equal -1.0 (sentinel for null last_transaction)."""
    return (vectors[:, 5] == -1.0) & (vectors[:, 6] == -1.0)


def boundary_oversample(
    vectors: np.ndarray, labels: np.ndarray, sample_size: int = 50_000
) -> tuple[np.ndarray, np.ndarray]:
    """Return augmented (vectors, labels) with boundary vectors duplicated 3x."""
    n = len(labels)
    sample_idx = np.random.choice(n, min(sample_size, n), replace=False)
    sample = vectors[sample_idx]
    sample_labels = labels[sample_idx]

    sample_gpu = cp.asarray(sample, dtype=cp.float32)
    vectors_gpu = cp.asarray(vectors, dtype=cp.float32)

    nbrs = cuNearestNeighbors(
        n_neighbors=5, algorithm="brute", metric="euclidean", output_type="numpy"
    )
    nbrs.fit(vectors_gpu)
    _, indices = nbrs.kneighbors(sample_gpu)

    fraud_counts = labels[indices].sum(axis=1)
    boundary_mask = (fraud_counts == 2) | (fraud_counts == 3)
    boundary_vecs = sample[boundary_mask]
    boundary_labs = sample_labels[boundary_mask]

    print(f"  Boundary vectors (fc=2 or 3): {boundary_mask.sum()} → duplicated 3×")

    vectors_aug = np.concatenate([vectors, boundary_vecs, boundary_vecs, boundary_vecs])
    labels_aug = np.concatenate([labels, boundary_labs, boundary_labs, boundary_labs])
    return vectors_aug, labels_aug


def two_level_kmeans(
    vectors: np.ndarray, labels: np.ndarray, vectors_aug: np.ndarray, K1: int, K2: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit 2-level KMeans on augmented vectors, assign original vectors.
    Returns: macro_centroids (K1×DIMS), micro_centroids (K1*K2×DIMS),
             micro_assignments (N,) — flat leaf index for each original vector.
    """
    N = len(labels)
    vectors_gpu = cp.asarray(vectors, dtype=cp.float32)
    vectors_gpu_aug = cp.asarray(vectors_aug, dtype=cp.float32)

    print(f"  Level-1 KMeans: K1={K1}, fit on {len(vectors_aug)} aug vecs...")
    km1 = KMeans(
        n_clusters=K1,
        init="scalable-k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
        output_type="numpy",
    )
    km1.fit(vectors_gpu_aug)
    macro_centroids = km1.cluster_centers_.astype(np.float32)  # (K1, DIMS)

    macro_assignments_orig = km1.predict(vectors_gpu).astype(np.int32)  # (N,)

    micro_centroids = np.zeros((K1, K2, DIMS), dtype=np.float32)
    micro_assignments = np.zeros(N, dtype=np.int32)

    macro_assignments_aug = km1.predict(vectors_gpu_aug).astype(np.int32)

    for i in range(K1):
        mask_aug = macro_assignments_aug == i
        vecs_aug_i = vectors_gpu_aug[mask_aug]
        if len(vecs_aug_i) < K2:
            # degenerate macro cluster: assign all to micro 0
            micro_assignments[macro_assignments_orig == i] = i * K2
            continue
        km2 = KMeans(
            n_clusters=K2,
            init="scalable-k-means++",
            n_init=5,
            max_iter=200,
            random_state=42,
            output_type="numpy",
        )
        km2.fit(vecs_aug_i)
        micro_centroids[i] = km2.cluster_centers_

        mask_orig = macro_assignments_orig == i
        orig_in_macro = cp.asarray(vectors[mask_orig], dtype=cp.float32)
        if len(orig_in_macro) > 0:
            local_assign = km2.predict(orig_in_macro).astype(np.int32)
            micro_assignments[mask_orig] = i * K2 + local_assign

        if (i + 1) % 16 == 0:
            print(f"    macro {i + 1}/{K1} done")

    micro_centroids_flat = micro_centroids.reshape(K1 * K2, DIMS)
    return macro_centroids, micro_centroids_flat, micro_assignments


def balanced_kmeans(
    vectors: np.ndarray,
    micro_centroids_flat: np.ndarray,
    micro_assignments: np.ndarray,
    K1: int,
    K2: int,
) -> np.ndarray:
    """Reassign overflow vectors (>1.5× avg leaf size) to nearest sibling with capacity."""
    N_leaves = K1 * K2
    cluster_sizes = np.bincount(micro_assignments, minlength=N_leaves).astype(np.int32)
    avg_size = len(vectors) / N_leaves
    max_size = int(avg_size * 1.5)
    print(f"  Balanced k-means: avg={avg_size:.0f}, cap={max_size}")

    for macro_id in range(K1):
        for micro_local in range(K2):
            c = macro_id * K2 + micro_local
            if cluster_sizes[c] <= max_size:
                continue
            idxs = np.where(micro_assignments == c)[0]
            cent = micro_centroids_flat[c]
            dists = np.linalg.norm(vectors[idxs] - cent, axis=1)
            n_overflow = cluster_sizes[c] - max_size
            overflow_idxs = idxs[np.argsort(-dists)[:n_overflow]]

            for v in overflow_idxs:
                siblings = [
                    macro_id * K2 + j
                    for j in range(K2)
                    if j != micro_local and cluster_sizes[macro_id * K2 + j] < max_size
                ]
                if not siblings:
                    continue
                sib_cents = micro_centroids_flat[[s for s in siblings]]
                nearest_sib = siblings[
                    int(np.argmin(np.linalg.norm(sib_cents - vectors[v], axis=1)))
                ]
                micro_assignments[v] = nearest_sib
                cluster_sizes[c] -= 1
                cluster_sizes[nearest_sib] += 1

    return micro_assignments


def compute_dsafe(vectors: np.ndarray, labels: np.ndarray) -> float:
    """
    D_safe = 99th percentile of L2 dist-to-5th-neighbor for samples whose
    brute-force k=5 gives fraudCount==0. Stored as L2 (not L2²).
    """
    legit_idxs = np.where(labels == 0)[0]
    sample_idx = legit_idxs[: min(10_000, len(legit_idxs))]
    sample_vecs = cp.asarray(vectors[sample_idx], dtype=cp.float32)

    nbrs = cuNearestNeighbors(
        n_neighbors=5, algorithm="brute", metric="euclidean", output_type="numpy"
    )
    nbrs.fit(cp.asarray(vectors, dtype=cp.float32))
    dists_sq, neighbor_idx = nbrs.kneighbors(sample_vecs)
    # dists_sq shape: (sample_size, 5) — squared euclidean from cuML

    neighbor_labels = labels[neighbor_idx]
    fraud_counts = neighbor_labels.sum(axis=1)
    truly_legit = fraud_counts == 0

    max_dists = np.sqrt(dists_sq[truly_legit, 4])  # L2 to 5th neighbor
    d_safe = float(np.percentile(max_dists, 99))
    print(
        f"  D_safe = {d_safe:.6f} (99th pct of dist-to-5th-neighbor for fraudCount==0)"
    )
    return d_safe


def build_ivfh(
    vectors: np.ndarray, labels: np.ndarray, K1: int, K2: int, dst: Path
) -> None:
    N = len(labels)
    print(f"Building IVFH: N={N}, K1={K1}, K2={K2} → {dst.name}")

    # Boundary oversampling
    vectors_aug, _ = boundary_oversample(vectors, labels)

    # 2-level KMeans
    macro_centroids, micro_centroids_flat, micro_assignments = two_level_kmeans(
        vectors, labels, vectors_aug, K1, K2
    )

    # Balanced k-means post-processing
    micro_assignments = balanced_kmeans(
        vectors, micro_centroids_flat, micro_assignments, K1, K2
    )

    # Sort vectors by leaf assignment
    sort_idx = np.argsort(micro_assignments, kind="stable")
    vectors_sorted = vectors[sort_idx]
    labels_sorted = labels[sort_idx]
    micro_assignments_sorted = micro_assignments[sort_idx]

    N_leaves = K1 * K2
    cluster_sizes = np.bincount(micro_assignments_sorted, minlength=N_leaves).astype(
        np.uint32
    )
    cluster_starts = np.zeros(N_leaves, dtype=np.uint32)
    cluster_starts[1:] = np.cumsum(cluster_sizes[:-1])

    # Cluster radii: max L2 dist from centroid to any vector in cluster
    cluster_radius = np.zeros(N_leaves, dtype=np.float32)
    for c in range(N_leaves):
        s, sz = int(cluster_starts[c]), int(cluster_sizes[c])
        if sz == 0:
            continue
        vecs_in_c = vectors_sorted[s : s + sz]
        cent = micro_centroids_flat[c]
        dists = np.linalg.norm(vecs_in_c - cent, axis=1)
        cluster_radius[c] = float(dists.max())

    # D_safe
    d_safe = compute_dsafe(vectors, labels)

    # Quantize vectors to int16
    vectors_int16 = np.clip(np.round(vectors_sorted * SCALE), -32768, 32767).astype(
        np.int16
    )

    # Bounding boxes in int16 space — exact bounds over stored vectors per micro-cluster
    box_min_int16 = np.zeros((N_leaves, DIMS), dtype=np.int16)
    box_max_int16 = np.zeros((N_leaves, DIMS), dtype=np.int16)
    for c in range(N_leaves):
        s, sz = int(cluster_starts[c]), int(cluster_sizes[c])
        if sz == 0:
            continue
        vecs_in_c = vectors_int16[s : s + sz]
        box_min_int16[c] = vecs_in_c.min(axis=0)
        box_max_int16[c] = vecs_in_c.max(axis=0)

    # Write IVFH binary
    dst.parent.mkdir(exist_ok=True)
    with open(dst, "wb") as out:
        out.write(b"IVFH")
        out.write(struct.pack("<f", d_safe))
        out.write(struct.pack("<II", K1, K2))
        out.write(struct.pack("<I", N))
        out.write(macro_centroids.astype("<f4").tobytes())
        out.write(micro_centroids_flat.astype("<f4").tobytes())
        out.write(cluster_starts.astype("<u4").tobytes())
        out.write(cluster_sizes.astype("<u4").tobytes())
        out.write(cluster_radius.astype("<f4").tobytes())
        out.write(box_min_int16.astype("<i2").tobytes())
        out.write(box_max_int16.astype("<i2").tobytes())
        out.write(vectors_int16.astype("<i2").tobytes())
        out.write(labels_sorted.astype("u1").tobytes())

    size_mb = dst.stat().st_size / 1024 / 1024
    fraud_pct = labels.mean() * 100
    print(f"  → {dst} ({size_mb:.1f} MB), fraud={fraud_pct:.1f}%, D_safe={d_safe:.4f}")


def main():
    root = Path(__file__).parent.parent
    src = root / "resources" / "references.json.gz"
    dst_dir = root / "index"

    print("Loading records...")
    with gzip.open(src) as f:
        records = json.load(f)
    n = len(records)
    print(f"Loaded {n} records")

    # Pad 14-dim features to 16 dims (zeros in dims 14, 15)
    vectors = np.zeros((n, DIMS), dtype=np.float32)
    vectors[:, :14] = np.array([rec["vector"] for rec in records], dtype=np.float32)
    labels = np.array(
        [1 if rec["label"] == "fraud" else 0 for rec in records], dtype=np.uint8
    )

    # Split by null last_transaction sentinel (dims 5 and 6 == -1.0)
    null_mask = detect_null_tx_mask(vectors)
    vectors_first, labels_first = vectors[null_mask], labels[null_mask]
    vectors_subseq, labels_subseq = vectors[~null_mask], labels[~null_mask]
    print(
        f"Split: first_tx={len(labels_first)} ({null_mask.mean() * 100:.1f}%), "
        f"subsequent_tx={len(labels_subseq)}"
    )

    # Split first_tx by unknown_merchant (dim 11): homogeneous sub-spaces → better IVF accuracy
    first_unknown_mask = vectors_first[:, DIM_UNKNOWN_MERCHANT] == 1.0

    partitions_first = [
        ("first_known",   ~first_unknown_mask, K2_FIRST, "first_known.ivfh"),
        ("first_unknown",  first_unknown_mask, K2_FIRST, "first_unknown.ivfh"),
    ]
    for name, mask, k2, filename in partitions_first:
        vp = vectors_first[mask]
        lp = labels_first[mask]
        k1 = choose_k1(len(lp), k2)
        print(f"Partition {name}: N={len(lp)} ({mask.mean()*100:.1f}%), K1={k1}, K2={k2}")
        build_ivfh(vp, lp, k1, k2, dst_dir / filename)

    # Split subseq by unknown_merchant (dim 11): same rationale
    subseq_unknown_mask = vectors_subseq[:, DIM_UNKNOWN_MERCHANT] == 1.0

    partitions_subseq = [
        ("subseq_known",    ~subseq_unknown_mask, K2_SUBSEQ, "subseq_known.ivfh"),
        ("subseq_unknown",   subseq_unknown_mask, K2_SUBSEQ, "subseq_unknown.ivfh"),
    ]
    for name, mask, k2, filename in partitions_subseq:
        vp = vectors_subseq[mask]
        lp = labels_subseq[mask]
        k1 = choose_k1(len(lp), k2)
        print(f"Partition {name}: N={len(lp)} ({mask.mean()*100:.1f}%), K1={k1}, K2={k2}")
        build_ivfh(vp, lp, k1, k2, dst_dir / filename)


if __name__ == "__main__":
    main()
