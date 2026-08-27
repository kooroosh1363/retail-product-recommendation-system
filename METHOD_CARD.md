# Recommendation Method Card

## Intended use
Educational/portfolio demonstration of leakage-aware offline Top-K retail recommendation.

## Feedback signal
Purchases are converted to implicit user-item interactions. The task is ranking future purchased items, not predicting quantities or ratings.

## Temporal validation
A global forward-only split is used. The last 28 days form the final test window and the preceding 28 days form validation. Candidate models are selected only on validation. The selected model is then refit on train+validation and evaluated once on test.

## Candidate methods
- overall popularity
- recent popularity
- item-item cosine collaborative filtering
- TruncatedSVD latent-factor recommender

## Evaluation
Top-K is K=10. Seen training items are excluded from recommendations. Metrics include Precision@10, Recall@10, Hit Rate@10, NDCG@10, catalog coverage, and cold-start share.

Model selection uses validation NDCG@10 with Recall@10 as a tie-breaker.

## Cold start
Interactions involving users or items unseen in the fitted catalog cannot be ranked by collaborative models. They are reported separately rather than silently counted as ordinary ranking failures.

## Limitations
- offline purchase replay is not online recommendation behavior;
- no impressions or exposure logs exist, so non-purchases are not verified dislikes;
- popularity bias can inflate performance on head items;
- no product content, margin, stock, promotion, or session context is used;
- global temporal windows may exclude users without activity in the future windows;
- coverage and ranking metrics do not prove business lift;
- recommendations are predictive, not causal.

## Production extensions
A production system would add impression/click logs, candidate generation + reranking, calibrated exploration, diversity/novelty constraints, inventory and margin features, session context, online A/B tests, drift monitoring, and explicit cold-start fallbacks.
