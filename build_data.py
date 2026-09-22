"""
Fraud & Risk Analytics Dashboard — data pipeline
Dataset: Kaggle "Credit Card Fraud Detection" (mlg-ulb), 284,807 transactions, 492 fraud.
Trains models once and exports everything the dashboard needs to data.json.

Design choices (all stated in the dashboard):
  * Exact duplicate rows are dropped BEFORE splitting so the same row can't sit in train and test.
  * 70/30 stratified train/test split (seed 42). Test set is never used for fitting or threshold choice.
  * Cost-optimal threshold is chosen on 5-fold out-of-fold predictions from the TRAIN set,
    then evaluated on the TEST set.
  * Segments (hour, amount tier, k-means cluster) are descriptive and use the full dataset.
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, silhouette_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

SEED = 42
rng = np.random.default_rng(SEED)
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data.json"

df = pd.read_csv(SRC)
V = [f"V{i}" for i in range(1, 29)]
out = {"meta": {}}

# ---------------------------------------------------------------- engineered segments
df["hour"] = ((df["Time"] % 86400) // 3600).astype(int)
df["day"] = (df["Time"] // 86400).astype(int) + 1
TOD = [("Overnight", 0, 6), ("Morning", 6, 12), ("Afternoon", 12, 18), ("Evening", 18, 24)]
df["tod"] = pd.cut(df["hour"], [0, 6, 12, 18, 24], right=False, labels=[t[0] for t in TOD]).astype(str)

AMT_EDGES = [0, 10, 50, 200, 1000, np.inf]
AMT_LABELS = ["Micro (<$10)", "Small ($10–50)", "Medium ($50–200)", "Large ($200–1K)", "Very large ($1K+)"]
df["amt_tier"] = pd.cut(df["Amount"], AMT_EDGES, right=False, labels=AMT_LABELS).astype(str)
qs = df["Amount"].quantile([.25, .5, .75, .9, .99]).round(2).to_dict()

# k-means on standardized V1–V28 (label NOT used)
Z = StandardScaler().fit_transform(df[V])
sample_idx = rng.choice(len(df), 20000, replace=False)
sil = {}
for k in range(4, 7):
    km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(Z[sample_idx])
    sil[k] = float(silhouette_score(Z[sample_idx], km.labels_, sample_size=10000, random_state=SEED))
K = max(sil, key=sil.get)
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
df["cluster_id"] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=V)
# order clusters by size so A is largest
order = df["cluster_id"].value_counts().index.tolist()
letters = "ABCDEF"
cluster_info = {}
for rank, cid in enumerate(order):
    c = cent.loc[cid]
    top = c.abs().sort_values(ascending=False).index[:2]
    # within-cluster spread vs global (std of standardized features = 1)
    spread = Z[df["cluster_id"].values == cid].std(axis=0)
    desc = " · ".join(f"{'high' if c[f] > 0 else 'low'} {f}" for f in top)
    cluster_info[cid] = {
        "name": f"Cluster {letters[rank]}",
        "desc": desc,
        "top": [{"f": f, "z": round(float(c[f]), 2)} for f in c.abs().sort_values(ascending=False).index[:5]],
        "spread": round(float(spread.mean()), 2),
    }
df["cluster"] = df["cluster_id"].map(lambda c: cluster_info[c]["name"])


def seg_table(col, order_list=None):
    g = df.groupby(col).agg(n=("Class", "size"), fraud=("Class", "sum"),
                            amt=("Amount", "sum"), famt=("Amount", lambda s: s[df.loc[s.index, "Class"] == 1].sum()))
    g["rate"] = g["fraud"] / g["n"]
    if order_list is not None:
        g = g.reindex(order_list)
    rows = []
    for k, r in g.iterrows():
        rows.append({"key": str(k), "n": int(r.n), "fraud": int(r.fraud), "rate": float(r.rate),
                     "amt": round(float(r.amt), 2), "famt": round(float(r.famt), 2)})
    return rows


clusters_seg = seg_table("cluster", [f"Cluster {letters[i]}" for i in range(K)])
for row in clusters_seg:
    info = next(v for v in cluster_info.values() if v["name"] == row["key"])
    row.update({"desc": info["desc"], "top": info["top"], "spread": info["spread"]})

out["segments"] = {
    "tod": seg_table("tod", [t[0] for t in TOD]),
    "amount": seg_table("amt_tier", AMT_LABELS),
    "cluster": clusters_seg,
}
out["meta"]["kmeans"] = {"k": K, "silhouette": {str(k): round(v, 3) for k, v in sil.items()}}
out["meta"]["amount_quantiles"] = {str(k): v for k, v in qs.items()}

# hour × amount tier heatmap + hourly marginals
hm = df.groupby(["hour", "amt_tier"]).agg(n=("Class", "size"), fraud=("Class", "sum")).reset_index()
out["heatmap"] = [{"h": int(r.hour), "t": AMT_LABELS.index(r.amt_tier), "n": int(r.n), "fraud": int(r.fraud)}
                  for r in hm.itertuples()]
hr = df.groupby("hour").agg(n=("Class", "size"), fraud=("Class", "sum")).reset_index()
out["hourly"] = [{"h": int(r.hour), "n": int(r.n), "fraud": int(r.fraud)} for r in hr.itertuples()]
out["amt_tiers"] = AMT_LABELS

# amount distribution (log bins), share within each class
bins = np.concatenate([[0, 0.5], np.logspace(0, np.log10(26000), 34)])
lc, _ = np.histogram(df.loc[df.Class == 0, "Amount"], bins)
fc, _ = np.histogram(df.loc[df.Class == 1, "Amount"], bins)
out["amount_hist"] = {"edges": [round(float(b), 2) for b in bins],
                      "legit": lc.tolist(), "fraud": fc.tolist()}
out["amount_stats"] = {
    "legit_median": float(df.loc[df.Class == 0, "Amount"].median()),
    "fraud_median": float(df.loc[df.Class == 1, "Amount"].median()),
    "legit_mean": float(df.loc[df.Class == 0, "Amount"].mean()),
    "fraud_mean": float(df.loc[df.Class == 1, "Amount"].mean()),
    "fraud_under_1": int(((df.Class == 1) & (df.Amount <= 1)).sum()),
}

# ---------------------------------------------------------------- modeling
n_raw = len(df)
dups = int(df.duplicated(subset=["Time"] + V + ["Amount", "Class"]).sum())
dm = df.drop_duplicates(subset=["Time"] + V + ["Amount", "Class"]).reset_index(drop=True)
dm["logamt"] = np.log1p(dm["Amount"])
FEATS = V + ["logamt", "hour"]
X, y, amt = dm[FEATS].values, dm["Class"].values, dm["Amount"].values
idx_tr, idx_te = train_test_split(np.arange(len(dm)), test_size=0.30, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
amt_tr, amt_te = amt[idx_tr], amt[idx_te]


def make_models():
    return {
        "xgb": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=1, eval_metric="aucpr",
                             random_state=SEED, n_jobs=8, tree_method="hist"),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
    }


# thresholds: dense near 0 where the cost optimum usually sits, then linear
TH = np.unique(np.round(np.concatenate([np.arange(0.001, 0.02, 0.001), np.arange(0.02, 0.1, 0.005),
                                        np.arange(0.1, 0.991, 0.01)]), 4))


def sweep(p, yy, aa):
    rows = []
    for t in TH:
        pred = p >= t
        tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
        fn = int((~pred & (yy == 1)).sum()); tn = int((~pred & (yy == 0)).sum())
        rows.append([float(t), tp, fp, fn, tn, round(float(aa[~pred & (yy == 1)].sum()), 2),
                     round(float(aa[pred & (yy == 1)].sum()), 2)])
    return rows  # t, tp, fp, fn, tn, fn_amount (lost), tp_amount (caught)


def downsample_curve(x, y_, tol=0.004):
    """Keep a point whenever the curve has moved more than `tol` (L1) since the last kept point."""
    pts = [(float(x[0]), float(y_[0]))]
    for a, b in zip(x[1:-1], y_[1:-1]):
        if abs(a - pts[-1][0]) + abs(b - pts[-1][1]) > tol:
            pts.append((float(a), float(b)))
    pts.append((float(x[-1]), float(y_[-1])))
    return [[round(a, 5), round(b, 5)] for a, b in pts]


models_out, test_scores = {}, {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, model in make_models().items():
    # out-of-fold predictions on TRAIN (for threshold selection)
    oof = np.zeros(len(ytr))
    for f_tr, f_va in skf.split(Xtr, ytr):
        m = make_models()[name].fit(Xtr[f_tr], ytr[f_tr])
        oof[f_va] = m.predict_proba(Xtr[f_va])[:, 1]
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    test_scores[name] = p
    fpr, tpr, _ = roc_curve(yte, p)
    prec, rec, _ = precision_recall_curve(yte, p)
    # bootstrap 95% CI for PR-AUC and ROC-AUC on test
    b_pr, b_roc = [], []
    for _ in range(300):
        bi = rng.integers(0, len(yte), len(yte))
        if yte[bi].sum() == 0: continue
        b_pr.append(average_precision_score(yte[bi], p[bi])); b_roc.append(roc_auc_score(yte[bi], p[bi]))
    imp = None
    if name == "xgb":
        g = model.get_booster().get_score(importance_type="gain")
        imp = sorted([{"f": FEATS[int(k[1:])] if k.startswith("f") else k, "v": float(v)} for k, v in g.items()],
                     key=lambda r: -r["v"])
        tot = sum(r["v"] for r in imp)
        for r in imp: r["v"] = round(r["v"] / tot, 4)
    else:
        coef = model.named_steps["logisticregression"].coef_[0]
        imp = sorted([{"f": f, "v": round(float(c), 4)} for f, c in zip(FEATS, coef)], key=lambda r: -abs(r["v"]))
    models_out[name] = {
        "label": "XGBoost (depth 4, 300 trees)" if name == "xgb" else "Logistic regression (L2, C=0.1)",
        "roc_auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
        "roc_auc_ci": [float(np.percentile(b_roc, 2.5)), float(np.percentile(b_roc, 97.5))],
        "pr_auc_ci": [float(np.percentile(b_pr, 2.5)), float(np.percentile(b_pr, 97.5))],
        "oof_pr_auc": float(average_precision_score(ytr, oof)),
        "roc": downsample_curve(fpr, tpr), "pr": downsample_curve(rec[::-1], prec[::-1]),
        "sweep_test": sweep(p, yte, amt_te), "sweep_oof": sweep(oof, ytr, amt_tr),
        "importance": imp[:15],
    }
    print(name, "ROC", round(models_out[name]["roc_auc"], 4), "PR", round(models_out[name]["pr_auc"], 4),
          models_out[name]["pr_auc_ci"])

out["models"] = models_out
out["thresholds"] = TH.tolist()
out["meta"].update({
    "n_raw": n_raw, "fraud_raw": int(df.Class.sum()), "amount_raw": round(float(df.Amount.sum()), 2),
    "fraud_amount_raw": round(float(df.loc[df.Class == 1, "Amount"].sum()), 2),
    "dups_removed": dups, "n_model": len(dm), "fraud_model": int(dm.Class.sum()),
    "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
    "fraud_train": int(ytr.sum()), "fraud_test": int(yte.sum()),
    "fraud_amount_test": round(float(amt_te[yte == 1].sum()), 2),
    "features": FEATS, "seed": SEED, "hours_span": round(float(df.Time.max() / 3600), 1),
})

# ---------------------------------------------------------------- live-feed replay sample (test set only)
te = dm.iloc[idx_te].copy()
te["p_xgb"], te["p_lr"] = test_scores["xgb"], test_scores["lr"]
fr = te[te.Class == 1]; lg = te[te.Class == 0].sample(1400, random_state=SEED)
feed = pd.concat([fr, lg]).sample(frac=1, random_state=SEED)
out["feed"] = [[int(i), round(float(r.Time), 0), round(float(r.Amount), 2), int(r.hour),
                round(float(r.p_xgb), 5), round(float(r.p_lr), 5), int(r.Class), r.cluster]
               for i, r in feed.iterrows()]
out["meta"]["feed_fraud_share"] = round(len(fr) / len(feed), 4)

with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("K", K, sil, "dups", dups, "test fraud", int(yte.sum()))
