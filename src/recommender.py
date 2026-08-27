from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_interactions, temporal_split

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
K = 10
RANDOM_STATE = 42


def build_matrix(train: pd.DataFrame):
    users = sorted(train["customer_id"].unique())
    items = sorted(train["item_id"].unique())
    uidx = {u:i for i,u in enumerate(users)}
    iidx = {it:i for i,it in enumerate(items)}
    rows = train["customer_id"].map(uidx).to_numpy()
    cols = train["item_id"].map(iidx).to_numpy()
    data = np.ones(len(train), dtype=float)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(len(users), len(items))).tocsr()
    mat.data = np.ones_like(mat.data)
    return mat, users, items, uidx, iidx


def popularity_scores(train: pd.DataFrame, items: list[str]) -> np.ndarray:
    counts = train.groupby("item_id")["customer_id"].nunique()
    return np.asarray([float(counts.get(it, 0.0)) for it in items])


def recent_popularity_scores(train: pd.DataFrame, items: list[str], days: int = 28) -> np.ndarray:
    cutoff = train["timestamp"].max() - pd.Timedelta(days=days)
    recent = train.loc[train["timestamp"] >= cutoff]
    counts = recent.groupby("item_id")["customer_id"].nunique()
    return np.asarray([float(counts.get(it, 0.0)) for it in items])


def item_similarity(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    sim = cosine_similarity(matrix.T, dense_output=False).tocsr()
    sim.setdiag(0.0)
    sim.eliminate_zeros()
    return sim


def svd_scores(matrix: sparse.csr_matrix, n_components: int = 48):
    n_components = min(n_components, max(2, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T
    return svd, user_factors, item_factors


def topk_from_scores(scores: np.ndarray, seen: np.ndarray, k: int = K) -> np.ndarray:
    x = np.asarray(scores, dtype=float).copy()
    x[seen] = -np.inf
    if len(x) <= k:
        return np.argsort(-x)
    idx = np.argpartition(-x, k)[:k]
    return idx[np.argsort(-x[idx])]


def evaluate(train: pd.DataFrame, future: pd.DataFrame, model_name: str) -> dict:
    matrix, users, items, uidx, iidx = build_matrix(train)
    catalog = set(items)
    future_known = future.loc[future["item_id"].isin(catalog) & future["customer_id"].isin(uidx)].copy()
    truths = future_known.groupby("customer_id")["item_id"].apply(lambda s: set(s)).to_dict()
    eligible = sorted(truths)
    if not eligible:
        raise ValueError("No eligible users for offline evaluation")

    pop = popularity_scores(train, items)
    recent_pop = recent_popularity_scores(train, items)
    sim = None
    svd = None
    user_factors = item_factors = None
    if model_name == "item_knn":
        sim = item_similarity(matrix)
    elif model_name == "svd":
        svd, user_factors, item_factors = svd_scores(matrix)

    hit_rates = []
    recalls = []
    precisions = []
    ndcgs = []
    recommended_items = set()

    for user in eligible:
        ui = uidx[user]
        seen = matrix[ui].indices
        if model_name == "popular":
            scores = pop
        elif model_name == "recent_popular":
            scores = recent_pop
        elif model_name == "item_knn":
            scores = np.asarray(matrix[ui].dot(sim).toarray()).ravel()
        elif model_name == "svd":
            scores = user_factors[ui].dot(item_factors.T)
        else:
            raise ValueError(model_name)

        rec_idx = topk_from_scores(scores, seen, K)
        recs = [items[i] for i in rec_idx if np.isfinite(scores[i])]
        recommended_items.update(recs)
        truth = truths[user]
        hits = [1 if it in truth else 0 for it in recs]
        n_hit = sum(hits)
        precisions.append(n_hit / K)
        recalls.append(n_hit / len(truth))
        hit_rates.append(1.0 if n_hit else 0.0)
        dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
        ideal_hits = min(len(truth), K)
        idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return {
        "model": model_name,
        "eligible_users": len(eligible),
        "precision_at_10": float(np.mean(precisions)),
        "recall_at_10": float(np.mean(recalls)),
        "hit_rate_at_10": float(np.mean(hit_rates)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "catalog_coverage_at_10": float(len(recommended_items) / len(items)),
        "known_future_rows": int(len(future_known)),
        "cold_start_future_rows": int(len(future) - len(future_known)),
        "cold_start_future_share": float(1 - len(future_known) / len(future)) if len(future) else 0.0,
    }


def main() -> None:
    ART.mkdir(exist_ok=True)
    interactions, audit = load_interactions()
    train, val, test, split_meta = temporal_split(interactions)

    candidates = ["popular", "recent_popular", "item_knn", "svd"]
    val_results = pd.DataFrame([evaluate(train, val, name) for name in candidates])
    selected = str(val_results.sort_values(["ndcg_at_10", "recall_at_10"], ascending=False).iloc[0]["model"])

    # After model choice is locked, refit on train+validation and evaluate once on test.
    train_val = pd.concat([train, val], ignore_index=True).sort_values("timestamp")
    test_result = evaluate(train_val, test, selected)

    matrix, users, items, _, _ = build_matrix(train_val)
    popularity = popularity_scores(train_val, items)
    artifact = {
        "selected_model": selected,
        "users": users,
        "items": items,
        "popularity": popularity,
    }
    if selected == "item_knn":
        artifact["item_similarity"] = item_similarity(matrix)
    elif selected == "svd":
        svd, uf, itf = svd_scores(matrix)
        artifact.update({"svd": svd, "user_factors": uf, "item_factors": itf})
    joblib.dump(artifact, ART / "model.joblib")

    val_results.to_csv(ART / "validation_metrics.csv", index=False)
    pd.DataFrame([test_result]).to_csv(ART / "test_metrics.csv", index=False)

    report = {
        "data_audit": audit,
        "split": split_meta,
        "evaluation_policy": {
            "top_k": K,
            "selection_metric": "validation NDCG@10 with Recall@10 tie-break",
            "candidate_models": candidates,
            "seen_item_policy": "items observed for the user in training are excluded from recommendations",
            "cold_start_policy": "future rows for unseen users/items are not rankable by collaborative models and are reported separately",
        },
        "validation_results": val_results.to_dict(orient="records"),
        "selected_model": selected,
        "test_result": test_result,
        "claim_boundary": "offline historical ranking evaluation; no guarantee of online CTR, conversion, revenue, or causal lift",
    }
    (ART / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
