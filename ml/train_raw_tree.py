# ml/train_raw_tree.py
"""
Train a decision tree on the normalized 14-dim reference vectors from references.json.gz.
Since raw transaction payloads are unavailable in the reference dataset, we train on the
same normalized vectors used by the IVF index. The resulting tree classifies based on
the same feature space as the k-NN search, enabling it to short-circuit confident cases.

The tree is evaluated using a feature subset that maps to RawFeatures fields via the
vectorizer's normalized dimensions. At inference time, the tree uses RawFeatures fields
that correspond to the same normalized dimensions.

Outputs ml/models/raw_tree.pkl with tree arrays ready for code generation.

Usage:
    uv run --with numpy --with scikit-learn ml/train_raw_tree.py

Notes:
  - The 14-dim normalized vector from references.json.gz corresponds to:
      [0]=amount_norm, [1]=installments_norm, [2]=amount_vs_avg, [3]=hour/23,
      [4]=day_of_week/6, [5]=minutes_since_last_tx (or -1), [6]=km_from_last_tx (or -1),
      [7]=km_from_home_norm, [8]=tx_count_24h_norm, [9]=is_online, [10]=card_present,
      [11]=unknown_merchant, [12]=mcc_risk, [13]=merchant_avg_amount_norm
  - We map these to the RawFeatures that can be derived without raw data at tree
    evaluation time. The tree is indexed by feature position in the 14-dim vector.
"""
import gzip
import json
import time
import pickle
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

CONFIDENCE_THRESHOLD = 0.85
MIN_SAMPLES_LEAF = 50

# Feature names matching the 14-dim normalized vector indices from references.json.gz.
# These correspond to the vectorizer output dimensions used in the IVF index.
FEATURE_NAMES = [
    "amount_norm",           # 0  clamp(amount / 10000)
    "installments_norm",     # 1  clamp(installments / 12)
    "amount_vs_avg",         # 2  clamp((amount / avg_amount) / 10)
    "hour_of_day_norm",      # 3  hour / 23
    "day_of_week_norm",      # 4  (weekday+6)%7 / 6
    "minutes_since_last_tx", # 5  clamp(minutes / 1440) or -1
    "km_from_last_tx",       # 6  clamp(km_from_current / 1000) or -1
    "km_from_home_norm",     # 7  clamp(km_from_home / 1000)
    "tx_count_24h_norm",     # 8  clamp(tx_count_24h / 20)
    "is_online",             # 9  0 or 1
    "card_present",          # 10 0 or 1
    "unknown_merchant",      # 11 1=unknown, 0=known
    "mcc_risk",              # 12 from mcc_risk.json
    "merchant_avg_amount_norm", # 13 clamp(merchant_avg_amount / 10000)
]


def _traverse(x, left, right, feature, threshold):
    node = 0
    while left[node] != -1:
        if x[feature[node]] <= threshold[node]:
            node = left[node]
        else:
            node = right[node]
    return node


def main():
    root = Path(__file__).parent.parent
    src = root / "resources" / "references.json.gz"
    out_dir = root / "ml" / "models"
    out_dir.mkdir(exist_ok=True)
    dst = out_dir / "raw_tree.pkl"

    print("Loading records...")
    t0 = time.time()
    with gzip.open(src) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records in {time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.array([r["vector"] for r in records], dtype=np.float32)
    y = np.array([1 if r["label"] == "fraud" else 0 for r in records], dtype=np.int8)
    print(f"Array construction: {time.time()-t0:.1f}s")
    print(f"X shape: {X.shape}, fraud rate: {y.mean():.3f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print("Training DecisionTreeClassifier...")
    t0 = time.time()
    clf = DecisionTreeClassifier(max_depth=None, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=42)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    print(f"Tree: {n_nodes} nodes, trained in {time.time()-t0:.1f}s")

    tree = clf.tree_
    n = tree.node_count
    left = tree.children_left.tolist()
    right = tree.children_right.tolist()
    feature = tree.feature.tolist()
    threshold = tree.threshold.tolist()
    values = tree.value  # shape: (n_nodes, 1, n_classes)

    confident = []
    leaf_class = []
    for i in range(n):
        if left[i] == -1:  # leaf
            counts = values[i, 0]
            total = counts.sum()
            majority_class = int(np.argmax(counts))
            confidence = counts[majority_class] / total if total > 0 else 0
            confident.append(confidence >= CONFIDENCE_THRESHOLD)
            leaf_class.append(majority_class)
        else:
            confident.append(False)
            leaf_class.append(0)

    covered = sum(1 for xi in X_val if confident[_traverse(xi, left, right, feature, threshold)])
    coverage_pct = covered / len(X_val) * 100

    fp = fn = 0
    for xi, yi in zip(X_val, y_val):
        node = _traverse(xi, left, right, feature, threshold)
        if confident[node]:
            pred = leaf_class[node]
            if pred == 1 and yi == 0:
                fp += 1
            elif pred == 0 and yi == 1:
                fn += 1

    print(f"\nValidation coverage: {coverage_pct:.1f}% of {len(X_val)} samples")
    print(f"FP (confident wrong legit): {fp}, FN (confident wrong fraud): {fn}")
    print(f"Confident leaves: {sum(confident)}")

    model = {
        "n_nodes": n,
        "children_left": left,
        "children_right": right,
        "feature": feature,
        "threshold": [float(x) for x in threshold],
        "confident": confident,
        "leaf_class": leaf_class,
        "feature_names": FEATURE_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "n_features": X.shape[1],
    }
    with open(dst, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
