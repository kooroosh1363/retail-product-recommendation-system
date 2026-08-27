# Data Source

This project uses the UCI Machine Learning Repository **Online Retail** dataset.

- Source: official UCI Online Retail archive
- Raw rows expected: 541,909
- Time span: December 2010 to December 2011
- Primary fields used: CustomerID, StockCode, InvoiceNo, InvoiceDate, Quantity, UnitPrice

The pipeline downloads the official ZIP at runtime and validates the raw row count before modeling.

## Cleaning scope

Rows with missing CustomerID, cancellation invoices, non-positive quantity/price, and exact duplicates are removed before implicit-feedback modeling.
