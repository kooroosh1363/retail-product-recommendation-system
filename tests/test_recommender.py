from pathlib import Path
import json
import pandas as pd

from src.recommender import main


def test_recommender_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "artifacts" / "metrics.json").read_text())

    assert metrics["evaluation_policy"]["top_k"] == 10
    assert metrics["selected_model"] in {"popular", "recent_popular", "item_knn", "svd"}
    test = metrics["test_result"]
    assert test["eligible_users"] > 100
    assert 0 <= test["precision_at_10"] <= 1
    assert 0 <= test["recall_at_10"] <= 1
    assert 0 <= test["hit_rate_at_10"] <= 1
    assert 0 <= test["ndcg_at_10"] <= 1
    assert 0 <= test["catalog_coverage_at_10"] <= 1
    assert 0 <= test["cold_start_future_share"] <= 1

    val = pd.read_csv(root / "artifacts" / "validation_metrics.csv")
    assert set(val["model"]) == {"popular", "recent_popular", "item_knn", "svd"}
    assert (root / "artifacts" / "test_metrics.csv").exists()
    assert (root / "artifacts" / "model.joblib").exists()
