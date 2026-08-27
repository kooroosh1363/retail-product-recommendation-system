from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import io

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_XLSX = RAW_DIR / "Online Retail.xlsx"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXPECTED_RAW_ROWS = 541_909


def download_raw() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XLSX.exists():
        return RAW_XLSX
    r = requests.get(UCI_ZIP_URL, timeout=120)
    r.raise_for_status()
    with ZipFile(io.BytesIO(r.content)) as zf:
        files = [x for x in zf.namelist() if x.lower().endswith('.xlsx')]
        if len(files) != 1:
            raise ValueError(f"Expected one XLSX, found {files}")
        RAW_XLSX.write_bytes(zf.read(files[0]))
    return RAW_XLSX


def load_interactions() -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(download_raw(), engine="openpyxl")
    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(f"Unexpected raw row count: {len(df)}")
    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    if not required.issubset(df.columns):
        raise ValueError("Unexpected schema")

    audit = {"raw_rows": int(len(df))}
    work = df.copy()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="raise")
    work["InvoiceNo"] = work["InvoiceNo"].astype(str)

    missing_customer = work["CustomerID"].isna()
    audit["missing_customer_rows_removed"] = int(missing_customer.sum())
    work = work.loc[~missing_customer].copy()

    cancelled = work["InvoiceNo"].str.startswith("C", na=False)
    audit["cancelled_rows_removed"] = int(cancelled.sum())
    work = work.loc[~cancelled].copy()

    positive = (work["Quantity"] > 0) & (work["UnitPrice"] > 0)
    audit["non_positive_rows_removed"] = int((~positive).sum())
    work = work.loc[positive].copy()

    before = len(work)
    work = work.drop_duplicates().copy()
    audit["exact_duplicates_removed"] = int(before - len(work))

    work["customer_id"] = work["CustomerID"].astype("int64").astype(str)
    work["item_id"] = work["StockCode"].astype(str)
    work = work[["customer_id", "item_id", "InvoiceDate", "Quantity"]].rename(columns={"InvoiceDate": "timestamp"})
    work = work.sort_values("timestamp").reset_index(drop=True)
    audit.update({
        "clean_rows": int(len(work)),
        "customers": int(work["customer_id"].nunique()),
        "items": int(work["item_id"].nunique()),
        "first_timestamp": work["timestamp"].min().isoformat(),
        "last_timestamp": work["timestamp"].max().isoformat(),
    })
    return work, audit


def temporal_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # The public source terminates on 2011-12-09 at 12:50 rather than at a full
    # end-of-day boundary. Exclude that terminal calendar date so the final
    # offline evaluation window contains complete observed calendar days only.
    source_last = interactions["timestamp"].max()
    terminal_day = source_last.normalize()
    terminal_excluded = source_last.time() != pd.Timestamp(source_last.date()).time()
    modeling = interactions.copy()
    excluded_terminal_rows = 0
    if terminal_excluded:
        terminal_mask = modeling["timestamp"].dt.normalize().eq(terminal_day)
        excluded_terminal_rows = int(terminal_mask.sum())
        modeling = modeling.loc[~terminal_mask].copy()

    max_day = modeling["timestamp"].max().normalize()
    test_start = max_day - pd.Timedelta(days=27)
    val_start = test_start - pd.Timedelta(days=28)

    train = modeling.loc[modeling["timestamp"] < val_start].copy()
    val = modeling.loc[(modeling["timestamp"] >= val_start) & (modeling["timestamp"] < test_start)].copy()
    test = modeling.loc[modeling["timestamp"] >= test_start].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition")
    if not (train["timestamp"].max() < val["timestamp"].min() <= val["timestamp"].max() < test["timestamp"].min()):
        raise ValueError("Temporal split ordering failed")

    meta = {
        "source_last_timestamp": source_last.isoformat(),
        "terminal_day_excluded_as_potentially_incomplete": bool(terminal_excluded),
        "excluded_terminal_date": terminal_day.date().isoformat() if terminal_excluded else None,
        "excluded_terminal_rows": excluded_terminal_rows,
        "modeled_last_day": max_day.date().isoformat(),
        "validation_start": val_start.isoformat(),
        "test_start": test_start.isoformat(),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
    }
    return train, val, test, meta
