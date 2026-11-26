"""
============================================================
 train_gnn.py  -  Train the graph-based fake-account detector
============================================================
Run once (takes a few minutes on the full dataset):

    python train_gnn.py                      # auto-finds twitter_combined.txt
    python train_gnn.py "D:\\GNN Mini project\\twitter_combined.txt"

What it does
  1. Loads the Twitter follower graph (edge list: "follower followee").
  2. Injects SYNTHETIC bot accounts (the SNAP data has no spam labels).
       - classic bots : follow 100-800 accounts, almost no followers/follow-backs
       - stealth bots : follow 20-120 accounts, a few followers (harder to catch)
       - all bots also follow a few other bots (bot-ring structure)
  3. Computes node features (degree, ratio, reciprocity, PageRank).
  4. Builds GraphSAGE-style embeddings: each node = [own features,
     mean of followees, mean of followers], applied for 2 hops.
  5. Trains an MLP classifier on the embeddings and compares it with
     feature-only baselines (Random Forest, Logistic Regression).
  6. Saves everything to model_bundle.joblib for the Streamlit app.

No PyTorch needed - only numpy / scipy / scikit-learn.
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
BOT_FRACTION = 0.05          # bots injected = 5% of real users
PROJECT_DIR = r"D:\GNN Mini project"
BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.joblib")
FEATURE_NAMES = ["log_following", "log_followers", "following_follower_gap",
                 "reciprocity", "pagerank_score"]


def find_dataset():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(PROJECT_DIR, "twitter_combined.txt"),
              os.path.join(PROJECT_DIR, "dataset", "twitter_combined.txt"),
              os.path.join(here, "twitter_combined.txt"),
              "twitter_combined.txt"]:
        if os.path.exists(p):
            return p
    sys.exit("twitter_combined.txt not found. Pass the path:  python train_gnn.py <path>")


# ------------------------------------------------------------------ graph
def load_graph(path):
    edges = pd.read_csv(path, sep=r"\s+", header=None, names=["src", "dst"], dtype=np.int64)
    edges = edges[edges.src != edges.dst].drop_duplicates()
    node_ids = np.unique(np.concatenate([edges.src.values, edges.dst.values]))
    s = np.searchsorted(node_ids, edges.src.values)
    d = np.searchsorted(node_ids, edges.dst.values)
    n = len(node_ids)
    A = sp.csr_matrix((np.ones(len(s), dtype=np.float32), (s, d)), shape=(n, n))
    return node_ids, A          # A[i, j] = 1  ->  node i follows node j


def inject_bots(A, rng):
    n = A.shape[0]
    nb = max(100, int(BOT_FRACTION * n))
    total = n + nb
    in_deg = np.asarray(A.sum(0)).ravel()
    pop_p = (in_deg + 1) / (in_deg + 1).sum()
    bot_ids = np.arange(n, total)
    rows, cols = [], []
    for b in bot_ids:
        stealth = rng.random() < 0.30
        k = int(rng.integers(20, 120)) if stealth else int(rng.integers(100, 800))
        k = max(1, min(k, n // 2))
        if rng.random() < 0.6:
            targets = rng.choice(n, k, replace=False)
        else:
            targets = rng.choice(n, k, replace=False, p=pop_p)
        rows.append(np.full(k, b)); cols.append(targets)                 # bot follows targets
        fb = rng.random(k) < (0.08 if stealth else 0.02)                 # rare follow-backs
        rows.append(targets[fb]); cols.append(np.full(int(fb.sum()), b))
        nf = int(rng.integers(0, 25 if stealth else 8))                  # few real followers
        srcs = rng.choice(n, min(nf, n), replace=False)
        rows.append(srcs); cols.append(np.full(len(srcs), b))
        peers = rng.choice(bot_ids, 4, replace=False)                    # bot ring
        peers = peers[peers != b]
        rows.append(np.full(len(peers), b)); cols.append(peers)
    A = A.tocoo()
    r = np.concatenate([A.row] + rows)
    c = np.concatenate([A.col] + cols)
    M = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(total, total))
    M.data[:] = 1.0
    y = np.zeros(total, dtype=int)
    y[n:] = 1
    return M, y


def pagerank(A, d=0.85, iters=40):
    n = A.shape[0]
    out = np.asarray(A.sum(1)).ravel()
    inv = np.where(out > 0, 1.0 / np.maximum(out, 1), 0.0)
    P_T = (sp.diags(inv) @ A).T.tocsr()
    dangling = out == 0
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = d * (P_T @ r) + (d * r[dangling].sum() + (1 - d)) / n
    return r


def node_features(A):
    n = A.shape[0]
    out = np.asarray(A.sum(1)).ravel()
    inn = np.asarray(A.sum(0)).ravel()
    mutual = np.asarray(A.multiply(A.T).sum(1)).ravel()
    recip = mutual / np.maximum(out, 1)
    pr = pagerank(A)
    prf = np.log10(pr * n)
    X = np.column_stack([np.log1p(out), np.log1p(inn),
                         np.log1p(out) - np.log1p(inn), recip, prf]).astype(np.float32)
    return X, pr, out


def row_norm(M):
    d = np.asarray(M.sum(1)).ravel()
    return (sp.diags(1.0 / np.maximum(d, 1)) @ M).tocsr()


def sage_layer(X, M_out, M_in):
    """[self || mean(followees) || mean(followers)]"""
    return np.hstack([X, M_out @ X, M_in @ X]).astype(np.float32)


# ------------------------------------------------------------------ eval
def evaluate(name, y_true, prob):
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, prob)
    step = max(1, len(fpr) // 200)
    return {
        "name": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob),
        "fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist(),
        "cm": confusion_matrix(y_true, pred).tolist(),
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    path = find_dataset()
    print(f"[1/6] Loading graph: {path}")
    node_ids, A_real = load_graph(path)
    n_real = A_real.shape[0]
    print(f"      {n_real:,} users, {A_real.nnz:,} follow edges")

    print("[2/6] Injecting synthetic bot accounts ...")
    A, y = inject_bots(A_real, rng)
    n_total = A.shape[0]
    print(f"      +{n_total - n_real:,} bots  ->  {n_total:,} nodes")

    print("[3/6] Computing node features ...")
    X, pr, outdeg = node_features(A)

    print("[4/6] GraphSAGE-style 2-hop aggregation ...")
    M_out, M_in = row_norm(A), row_norm(A.T.tocsr())
    H1 = sage_layer(X, M_out, M_in)       # 15 dims
    H2 = sage_layer(H1, M_out, M_in)      # 45 dims

    idx = np.arange(n_total)
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)

    print("[5/6] Training models ...")
    scaler = StandardScaler().fit(H2[tr])
    gnn = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                        max_iter=300, random_state=SEED)
    gnn.fit(scaler.transform(H2[tr]), y[tr])
    p_gnn = gnn.predict_proba(scaler.transform(H2[te]))[:, 1]

    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED,
                                class_weight="balanced").fit(X[tr], y[tr])
    p_rf = rf.predict_proba(X[te])[:, 1]

    x_scaler = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(x_scaler.transform(X[tr]), y[tr])
    p_lr = lr.predict_proba(x_scaler.transform(X[te]))[:, 1]

    metrics = {
        "gnn": evaluate("GraphSAGE-style GNN (2-hop)", y[te], p_gnn),
        "rf": evaluate("Random Forest (own features only)", y[te], p_rf),
        "lr": evaluate("Logistic Regression (own features only)", y[te], p_lr),
        "loss_curve": [float(v) for v in gnn.loss_curve_],
        "rf_importance": dict(zip(FEATURE_NAMES, [float(v) for v in rf.feature_importances_])),
    }
    for k in ("gnn", "rf", "lr"):
        m = metrics[k]
        print(f"      {m['name']:<42} acc={m['accuracy']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}")

    print("[6/6] Saving bundle ...")
    gnn_prob = gnn.predict_proba(scaler.transform(H2[:n_real]))[:, 1].astype(np.float32)
    legit = X[:n_real]
    bundle = {
        "node_ids": node_ids, "A": A_real, "X": X, "H1": H1, "pr": pr, "outdeg": outdeg,
        "scaler": scaler, "gnn": gnn, "gnn_prob": gnn_prob,
        "mu": legit.mean(0), "sd": legit.std(0) + 1e-6,
        "metrics": metrics,
        "meta": {"n_real": n_real, "n_total": n_total, "n_bots": n_total - n_real,
                 "n_edges": int(A_real.nnz), "feature_names": FEATURE_NAMES},
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    print(f"Done in {time.time() - t0:.0f}s  ->  {BUNDLE_PATH}")
    print("Now run:  streamlit run spam_alert_app.py")


if __name__ == "__main__":
    main()
