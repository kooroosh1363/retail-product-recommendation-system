# Data Dictionary

## Raw source fields used
- `CustomerID`: customer identifier
- `StockCode`: product identifier
- `InvoiceNo`: invoice identifier; values beginning with `C` are cancellations
- `InvoiceDate`: interaction timestamp
- `Quantity`: purchased units
- `UnitPrice`: item unit price

## Modeled interaction fields
- `customer_id`: normalized string customer identifier
- `item_id`: normalized string product identifier
- `timestamp`: purchase timestamp
- `Quantity`: retained transaction quantity

## Evaluation outputs
- `precision_at_10`: relevant recommended items divided by 10
- `recall_at_10`: relevant recommended items divided by known future relevant items
- `hit_rate_at_10`: share of eligible users with at least one hit
- `ndcg_at_10`: position-discounted ranking quality
- `catalog_coverage_at_10`: share of train catalog appearing in at least one recommendation list
- `cold_start_future_share`: share of future rows involving a user or item not rankable from fitted history
