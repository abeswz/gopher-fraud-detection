# ml/train_decision_tree.py
"""
Train a decision tree classifier on the reference dataset.
Reads resources/references.json.gz (vectors already pre-computed).
Outputs ml/decision_tree_model.json for Go code generation.

Usage:
    uv run ml/train_decision_tree.py [--depth 20] [--min-leaf 50] [--confidence 0.95]
"""
import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20, help="max tree depth")
    ap.add_argument("--min-leaf", type=int, default=50, help="min samples per leaf")
    ap.add_argument("--confidence", type=float, default=0.95,
                    help="min majority class fraction to mark leaf as confident")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    src = root / "resources" / "references.json.gz"
    dst = root / "ml" / "decision_tree_model.json"

    print(f"Loading {src}...")
    with gzip.open(src) as f:
        records = json.load(f)

    n = len(records)
    print(f"Loaded {n} records")

    vectors = np.array([rec["vector"] for rec in records], dtype=np.float32)
    labels = np.array(
        [1 if rec["label"] == "fraud" else 0 for rec in records], dtype=np.int32
    )

    fraud_count = int(labels.sum())
    print(f"Labels: {fraud_count} fraud ({100*fraud_count/n:.1f}%), {n-fraud_count} legit")

    print(f"Training DecisionTreeClassifier(max_depth={args.depth}, min_samples_leaf={args.min_leaf})...")
    clf = DecisionTreeClassifier(
        max_depth=args.depth,
        min_samples_leaf=args.min_leaf,
        random_state=42,
    )
    clf.fit(vectors, labels)

    tree = clf.tree_
    n_nodes = tree.node_count
    print(f"Tree: {n_nodes} nodes, {tree.max_depth} depth")

    # Determine confidence for each node:
    # value[node] has shape [1, n_classes] = [[n_legit, n_fraud]]
    confident = []
    leaf_class = []
    for i in range(n_nodes):
        v = tree.value[i][0]
        total = float(v.sum())
        majority = float(v.max())
        frac = majority / total if total > 0 else 0.0
        is_leaf = tree.children_left[i] == -1
        is_confident = is_leaf and frac >= args.confidence
        confident.append(bool(is_confident))
        if is_leaf:
            leaf_class.append(int(np.argmax(v)))  # 0=legit, 1=fraud
        else:
            leaf_class.append(-1)

    conf_count = sum(confident)
    print(f"Confident leaves: {conf_count} of {n_nodes} nodes "
          f"(confidence threshold={args.confidence})")

    model = {
        "n_nodes": n_nodes,
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": [float(x) for x in tree.threshold],
        "confident": confident,
        "leaf_class": leaf_class,
    }

    with open(dst, "w") as f:
        json.dump(model, f)
    print(f"Model saved to {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
