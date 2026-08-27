# DS-06 — Retail Product Recommendation System

Portfolio-grade recommender-system project that converts historical retail purchases into a leakage-aware Top-K product ranking workflow.

## What this project demonstrates

- official UCI Online Retail acquisition and validation
- explicit cleaning audit for missing customers, cancellations, non-positive rows, and duplicates
- implicit user-item interaction modeling
- global forward-only temporal validation
- popularity and recent-popularity baselines
- item-item cosine collaborative filtering
- latent-factor recommendation with TruncatedSVD
- Top-10 ranking evaluation
- Precision@10, Recall@10, Hit Rate@10 and NDCG@10
- catalog coverage diagnostics
- explicit cold-start accounting
- seen-item exclusion
- model selection on validation only
- final untouched test evaluation
- reproducible model artifacts, tests and GitHub Actions CI

## Data

The project uses the UCI **Online Retail** dataset with 541,909 raw invoice-line rows covering December 2010 through December 2011.

Rows with missing CustomerID, cancellation invoices, non-positive quantity/price, and exact duplicates are removed before building implicit purchase interactions.

See `DATA_SOURCE.md`, `DATA_DICTIONARY.md`, and `METHOD_CARD.md` for provenance, field definitions, evaluation rules, and limitations.

## Architecture

```text
official UCI Online Retail ZIP
    -> raw row-count + schema validation
    -> clean customer purchase interactions
    -> normalize user/item identifiers
    -> global temporal split
         train
         validation: preceding 28 days
         test: final 28 days
    -> build sparse user-item matrix
    -> candidate recommenders
         overall popularity
         recent popularity
         item-item cosine CF
         TruncatedSVD latent factors
    -> remove items already seen in fitted history
    -> Top-10 ranking
    -> validation NDCG@10 + Recall@10 tie-break
    -> lock model choice
    -> refit on train + validation
    -> untouched final test
    -> ranking metrics + coverage + cold-start audit
    -> artifacts + pytest + GitHub Actions
```

## Why this is a ranking problem

The source contains purchases, not explicit star ratings. The project therefore treats purchases as **implicit positive feedback** and evaluates whether future purchased items are ranked near the top of each eligible customer's recommendation list.

A missing purchase is not treated as a verified dislike because the dataset does not contain impression/exposure logs.

## Candidate methods

- `popular`: overall unique-customer product popularity
- `recent_popular`: popularity within the recent 28-day training window
- `item_knn`: item-item cosine similarity from the sparse user-item matrix
- `svd`: TruncatedSVD latent-factor representation

Simple popularity baselines participate in model selection and are allowed to win.

## Leakage controls

Random splitting is intentionally not used.

The final 28 days are held out for test. The preceding 28 days are validation. Candidate models are selected using only the validation window. After model choice is locked, the selected method is refit on train plus validation and evaluated once on test.

Items already observed for a user in fitted history are removed from that user's recommendation list.

## Metrics

Top-K is fixed at `K=10`.

- **Precision@10** — fraction of the ten recommendations that are future relevant items
- **Recall@10** — fraction of known future relevant items recovered in the top ten
- **Hit Rate@10** — fraction of eligible users receiving at least one relevant recommendation
- **NDCG@10** — rewards relevant items appearing closer to the top of the ranking
- **Catalog Coverage@10** — share of the fitted catalog appearing in at least one recommendation list

Model selection uses validation **NDCG@10** with **Recall@10** as a tie-breaker.

## Cold-start policy

Future interactions involving an unseen user or unseen item cannot be ranked by a collaborative model trained only on past interactions. These rows are reported separately through `cold_start_future_share` instead of being silently mixed into normal ranking quality.

## Generated artifacts

Running the pipeline writes ignored outputs to `artifacts/`:

- `metrics.json`
- `validation_metrics.csv`
- `test_metrics.csv`
- `model.joblib`

## Claim boundary

This project demonstrates **offline historical ranking evaluation**. Offline ranking metrics do not guarantee online click-through rate, conversion, revenue, retention, or causal business lift.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.recommender
```
