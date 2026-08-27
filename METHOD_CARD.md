# Recommendation Method Card

## Intended use
Educational/portfolio demonstration of leakage-aware offline Top-K retail recommendation.

## Feedback signal
Purchases are converted to implicit user-item interactions. The task is ranking future purchased items, not predicting quantities or ratings.

## Temporal validation
A global forward-only split is used. Because the public source ends on 2011-12-09 at 12:50 rather than a full end-of-day boundary, that terminal calendar date is excluded from temporal evaluation. The final 28 complete calendar days form the test window and the preceding 28 complete calendar days form validation. Candidate models are selected only on validation. The selected model is then refit on train+validation and evaluated once on test.

## Candidate methods
- overall popularity
- recent popularity
- item-item cosine collaborative filtering
- TruncatedSVD latent-factor recommender

## Evaluation task
Top-K is K=10. Items already seen by a user in fitted history are excluded from recommendations.

To keep the target aligned with that policy, ranking metrics are computed on **future catalog items that the user has not previously seen**. Repeat purchases of already-seen items are not counted as impossible misses; they are reported separately as a repeat-purchase diagnostic.

Metrics include Precision@10, Recall@10, Hit Rate@10, NDCG@10, catalog coverage, repeat-seen share, and cold-start share.

Model selection uses validation NDCG@10 with Recall@10 as a tie-breaker.

## Cold start
Future rows are separated into:
- unseen-user rows;
- unseen-item rows for otherwise known users;
- known-user/known-item rows.

Rows involving unseen users or unseen items are not rankable by collaborative models trained only on fitted history and are reported separately rather than silently mixed into ordinary ranking quality.

## Limitations
- offline purchase replay is not online recommendation behavior;
- no impressions or exposure logs exist, so non-purchases are not verified dislikes;
- popularity bias can inflate performance on head items;
- no product content, margin, stock, promotion, or session context is used;
- global temporal windows may exclude users without activity in the future windows;
- the main ranking task is novel-item recommendation and does not evaluate repeat-purchase prediction;
- coverage and ranking metrics do not prove business lift;
- recommendations are predictive, not causal.

## Production extensions
A production system would add impression/click logs, separate repeat-purchase and discovery objectives, candidate generation + reranking, calibrated exploration, diversity/novelty constraints, inventory and margin features, session context, online A/B tests, drift monitoring, and explicit cold-start fallbacks.
