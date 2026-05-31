import argparse
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

N_CLUSTERS = 4000
NPROBE_DEFAULT = 20
LEAF_SIZE = 256


def build_ivf(vectors, labels, dst):
    n = len(labels)
    print(f"Fitting MiniBatchKMeans with {N_CLUSTERS} clusters...")
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=3,
        batch_size=100_000,
        verbose=0,
    )
    assignments = km.fit_predict(vectors)
    centroids = km.cluster_centers_.astype(np.float32)
    print(f"K-means done. Centroids: {centroids.shape}")

    sort_idx = np.argsort(assignments, kind="stable")
    vectors_sorted = vectors[sort_idx]
    labels_sorted = labels[sort_idx]

    cluster_sizes = np.bincount(assignments, minlength=N_CLUSTERS).astype(np.uint32)
    cluster_starts = np.zeros(N_CLUSTERS, dtype=np.uint32)
    cluster_starts[1:] = np.cumsum(cluster_sizes[:-1])

    vectors_int16 = np.clip(np.round(vectors_sorted * 10000), -32768, 32767).astype(
        np.int16
    )

    print("Writing IVF index...")
    with open(dst, "wb") as out:
        out.write(b"IVF1")
        out.write(struct.pack("<II", N_CLUSTERS, n))
        out.write(centroids.astype("<f4").tobytes())
        out.write(cluster_starts.astype("<u4").tobytes())
        out.write(cluster_sizes.astype("<u4").tobytes())
        out.write(vectors_int16.astype("<i2").tobytes())
        out.write(labels_sorted.astype("u1").tobytes())

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"{n} vectors, {N_CLUSTERS} clusters → {dst} ({size_mb:.1f} MB)")
    avg = cluster_sizes.mean()
    print(
        f"Avg cluster size: {avg:.0f}, nprobe={NPROBE_DEFAULT} → ~{avg * NPROBE_DEFAULT:.0f} vecs/query"
    )


def build_vptree(vectors, labels, dst):
    n = len(labels)
    rng = np.random.default_rng(42)
    vectors_int16 = np.clip(np.round(vectors * 10000), -32768, 32767).astype(np.int16)

    def build(indices):
        if len(indices) <= LEAF_SIZE:
            return ("leaf", indices)

        pivot_pos = int(rng.integers(len(indices)))
        pivot_idx = indices[pivot_pos]
        pivot_vec = vectors[pivot_idx]

        diffs = vectors[indices] - pivot_vec
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        dists = np.sqrt(dists_sq)
        tau = float(np.median(dists))

        mask = dists <= tau
        if mask.all() or (~mask).all():
            mid = len(indices) // 2
            inner = indices[:mid]
            outer = indices[mid:]
        else:
            inner = indices[mask]
            outer = indices[~mask]

        return ("node", pivot_idx, tau, build(inner), build(outer))

    nodes = []
    vec_order = []

    def serialize(tree):
        if tree[0] == "leaf":
            _, indices = tree
            vec_start = len(vec_order)
            vec_order.extend(indices.tolist())
            nodes.append(
                {
                    "leaf": True,
                    "childOff": vec_start,
                    "count": len(indices),
                    "tau": 0.0,
                    "vec": np.zeros(14, dtype=np.int16),
                }
            )
            return

        _, pivot_idx, tau, left, right = tree
        ni = len(nodes)
        nodes.append(None)
        serialize(left)
        right_ni = len(nodes)
        vec_i16 = np.clip(np.round(vectors[pivot_idx] * 10000), -32768, 32767).astype(
            np.int16
        )
        nodes[ni] = {
            "leaf": False,
            "tau": tau,
            "childOff": right_ni,
            "count": 0,
            "vec": vec_i16,
        }
        serialize(right)

    print("Building VP-tree (recursive)...")
    sys.setrecursionlimit(50000)
    all_indices = np.arange(n)
    tree = build(all_indices)
    print(f"Tree built. Serializing {n} vectors...")
    serialize(tree)

    vec_order_arr = np.array(vec_order, dtype=np.int64)
    vectors_dfs = vectors_int16[vec_order_arr]
    labels_dfs = labels[vec_order_arr]

    node_count = len(nodes)
    print(f"Writing VPT1 index ({node_count} nodes)...")
    with open(dst, "wb") as out:
        out.write(b"VPT1")
        out.write(struct.pack("<III", n, node_count, LEAF_SIZE))
        for nd in nodes:
            out.write(struct.pack("<fIHH", nd["tau"], nd["childOff"], nd["count"], 0))
            out.write(nd["vec"].astype("<i2").tobytes())
        out.write(vectors_dfs.astype("<i2").tobytes())
        out.write(labels_dfs.astype("u1").tobytes())

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"{n} vectors, {node_count} nodes → {dst} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build fraud detection search index")
    parser.add_argument(
        "--algo",
        choices=["vptree", "ivf"],
        default="vptree",
        help="Index algorithm (default: vptree)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    src = root / "resources" / "references.json.gz"
    dst = root / "index" / "references.bin"
    dst.parent.mkdir(exist_ok=True)

    print("Loading records...")
    with gzip.open(src) as f:
        records = json.load(f)

    n = len(records)
    print(f"Loaded {n} records")

    vectors = np.array([rec["vector"] for rec in records], dtype=np.float32)
    labels = np.array(
        [1 if rec["label"] == "fraud" else 0 for rec in records], dtype=np.uint8
    )

    if args.algo == "vptree":
        build_vptree(vectors, labels, dst)
    else:
        build_ivf(vectors, labels, dst)


if __name__ == "__main__":
    main()
