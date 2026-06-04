import gzip
import json
import struct
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.cluster import KMeans

N_CLUSTERS = 8000
N_INIT     = 10
NPROBE_DEFAULT = 15


def build_ivf(vectors, labels, dst):
    n = len(labels)
    vectors_16 = np.zeros((n, 16), dtype=np.float32)
    vectors_16[:, :14] = vectors
    vectors_gpu = cp.asarray(vectors_16, dtype=cp.float32)

    print(f"Fitting KMeans (cuML scalable-k-means++) with {N_CLUSTERS} clusters, n_init={N_INIT}...")
    km = KMeans(
        n_clusters=N_CLUSTERS,
        init='scalable-k-means++',
        n_init=N_INIT,
        max_iter=300,
        random_state=42,
        output_type='numpy',
    )
    km.fit(vectors_gpu)
    centroids   = km.cluster_centers_.astype(np.float32)  # (N_CLUSTERS, 16)
    assignments = km.labels_.astype(np.int32)             # (n,)
    print(f"KMeans done. Centroids: {centroids.shape}")

    sort_idx       = np.argsort(assignments, kind="stable")
    vectors_sorted = vectors_16[sort_idx]
    labels_sorted  = labels[sort_idx]

    cluster_sizes  = np.bincount(assignments, minlength=N_CLUSTERS).astype(np.uint32)
    cluster_starts = np.zeros(N_CLUSTERS, dtype=np.uint32)
    cluster_starts[1:] = np.cumsum(cluster_sizes[:-1])

    vectors_int16 = np.clip(
        np.round(vectors_sorted * 10000), -32768, 32767
    ).astype(np.int16)

    print("Writing IVF2 index...")
    with open(dst, "wb") as out:
        out.write(b"IVF2")
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


def main():
    root = Path(__file__).parent.parent
    src  = root / "resources" / "references.json.gz"
    dst  = root / "index" / "references.bin"
    dst.parent.mkdir(exist_ok=True)

    print("Loading records...")
    with gzip.open(src) as f:
        records = json.load(f)

    n = len(records)
    print(f"Loaded {n} records")

    vectors = np.array([rec["vector"] for rec in records], dtype=np.float32)
    labels  = np.array(
        [1 if rec["label"] == "fraud" else 0 for rec in records], dtype=np.uint8
    )

    build_ivf(vectors, labels, dst)


if __name__ == "__main__":
    main()
