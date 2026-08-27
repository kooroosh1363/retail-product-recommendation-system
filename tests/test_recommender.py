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

    split = metrics["split"]
    assert split["terminal_day_excluded_as_potentially_incomplete"] is True
    assert split["source_last_timestamp"] == "2011-12-09T12:50:00"
    assert split["excluded_terminal_date"] == "2011-12-09"
    assert split["excluded_terminal_rows"] > 0
    assert split["modeled_last_day"] == "2011-12-08"

    test = metrics["test_result"]
    assert test["eligible_users"] > 100
    assert 0 <= test["precision_at_10"] <= 1
    assert 0 <= test["recall_at_10"] <= 1
    assert 0 <= test["hit_rate_at_10"] <= 1
    assert 0 <= test["ndcg_at_10"] <= 1
    assert 0 <= test["catalog_coverage_at_10"] <= 1
    assert 0 <= test["cold_start_future_share"] <= 1
    assert 0 <= test["repeat_seen_future_share_of_known"] <= 1
    assert test["novel_rankable_future_rows"] + test["repeat_seen_future_rows"] == test["known_future_rows"]
    assert test["unseen_user_future_rows"] + test["unseen_item_future_rows_for_known_users"] == test["cold_start_future_rows"]

    val = pd.read_csv(root / "artifacts" / "validation_metrics.csv")
    assert set(val["model"]) == {"popular", "recent_popular", "item_knn", "svd"}
    assert (val["novel_rankable_future_rows"] > 0).all()
    assert (root / "artifacts" / "test_metrics.csv").exists()
    assert (root / "artifacts" / "model.joblib").exists()
