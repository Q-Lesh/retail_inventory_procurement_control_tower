from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------

def read_csv_auto(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        dtype=str,
    )

    if usecols is not None:
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns: {missing}")
        df = df[usecols].copy()

    LOG.info("Loaded %-28s shape=%s", path.name, df.shape)
    return df


def to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def normalize_bool01(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "y", "yes"}
    falsy = {"0", "false", "n", "no"}
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s.isin(truthy)] = 1.0
    out[s.isin(falsy)] = 0.0
    return out


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom2 = denom.replace({0: np.nan})
    return numer / denom2


# ----------------------------
# Core build
# ----------------------------

def build_supplier_scorecard(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    # dim_date: need Year + Date (for invoice delay)
    dim_date = read_csv_auto(input_dir / "dim_date.csv", usecols=["Date_ID", "Year", "Date"]).copy()
    dim_date["Date"] = pd.to_datetime(dim_date["Date"], errors="coerce")

    dim_supplier = read_csv_auto(
        input_dir / "dim_supplier.csv",
        usecols=[
            "Supplier_ID", "Supplier_Name", "Supplier_Category",
            "Region", "Country", "Reliability_Score", "Contract_Type"
        ],
    )

    fact_po = read_csv_auto(input_dir / "fact_purchase_orders.csv")
    fact_inv = read_csv_auto(input_dir / "fact_supplier_invoices.csv")

    # ----------------------------
    # PO block (Supplier_ID × Year)
    # ----------------------------

    po_cols = [
        "PO_ID", "PO_Line_ID", "Supplier_ID", "Date_ID_PO",
        "Order_Qty", "Received_Qty",
        "Lead_Time_Days_Planned", "Lead_Time_Days_Actual", "Is_Late"
    ]
    po_cols_exist = [c for c in po_cols if c in fact_po.columns]
    if "Supplier_ID" not in po_cols_exist or "Date_ID_PO" not in po_cols_exist:
        raise ValueError("fact_purchase_orders.csv must contain Supplier_ID and Date_ID_PO to build supplier scorecard.")

    po = fact_po[po_cols_exist].copy()
    to_numeric(po, ["Order_Qty", "Received_Qty", "Lead_Time_Days_Planned", "Lead_Time_Days_Actual"])

    # Attach Year from dim_date (Date_ID_PO)
    po = po.merge(
        dim_date[["Date_ID", "Year"]].rename(columns={"Date_ID": "Date_ID_PO"}),
        on="Date_ID_PO",
        how="left",
    )

    # Late flag normalization (accept 1/true/yes/etc.)
    if "Is_Late" in po.columns:
        po["Is_Late_Bin"] = normalize_bool01(po["Is_Late"])
    else:
        po["Is_Late_Bin"] = np.nan

    po_grp = (
        po.groupby(["Supplier_ID", "Year"], dropna=False)
        .agg(
            PO_Lines_Count=("PO_ID", "nunique"),
            PO_Units_Ordered_Year=("Order_Qty", "sum"),
            PO_Units_Received_Year=("Received_Qty", "sum"),
            PO_LT_Planned_Avg=("Lead_Time_Days_Planned", "mean"),
            PO_LT_Actual_Avg=("Lead_Time_Days_Actual", "mean"),
            PO_Late_Count=("Is_Late_Bin", lambda s: np.nansum(s.values)),
        )
        .reset_index()
    )

    po_grp["PO_FillRate_Year"] = safe_div(po_grp["PO_Units_Received_Year"], po_grp["PO_Units_Ordered_Year"])
    po_grp["PO_Pct_Late"] = safe_div(po_grp["PO_Late_Count"], po_grp["PO_Lines_Count"])

    # ----------------------------
    # Invoice block (Supplier_ID × Year)
    # ----------------------------

    inv_cols = [
        "Invoice_ID", "Supplier_ID",
        "Date_ID_Invoice", "Date_ID_Due", "Date_ID_Paid",
        "Invoiced_Amount"
    ]
    inv_cols_exist = [c for c in inv_cols if c in fact_inv.columns]
    if "Supplier_ID" not in inv_cols_exist or "Date_ID_Invoice" not in inv_cols_exist:
        raise ValueError("fact_supplier_invoices.csv must contain Supplier_ID and Date_ID_Invoice to build supplier scorecard.")

    inv = fact_inv[inv_cols_exist].copy()
    to_numeric(inv, ["Invoiced_Amount"])

    # Attach actual dates
    dd = dim_date[["Date_ID", "Date", "Year"]].copy()

    def add_date(df: pd.DataFrame, key_col: str, new_col: str) -> pd.DataFrame:
        if key_col in df.columns:
            return df.merge(dd.rename(columns={"Date_ID": key_col, "Date": new_col}), on=key_col, how="left")
        df[new_col] = pd.NaT
        return df

    inv = add_date(inv, "Date_ID_Invoice", "Invoice_Date")
    inv = add_date(inv, "Date_ID_Due", "Due_Date")
    inv = add_date(inv, "Date_ID_Paid", "Paid_Date")

    # Year by invoice date id
    inv = inv.merge(
        dd[["Date_ID", "Year"]].rename(columns={"Date_ID": "Date_ID_Invoice"}),
        on="Date_ID_Invoice",
        how="left",
    )

    inv["Payment_Delay_Days"] = (inv["Paid_Date"] - inv["Due_Date"]).dt.days
    inv["Is_Payment_Late"] = inv["Payment_Delay_Days"] > 0

    inv_grp = (
        inv.groupby(["Supplier_ID", "Year"], dropna=False)
        .agg(
            Invoices_Count=("Invoice_ID", "nunique"),
            Invoiced_Amount_Year=("Invoiced_Amount", "sum"),
            Payment_Delay_Days_Avg=("Payment_Delay_Days", "mean"),
            Invoices_Late_Count=("Is_Payment_Late", lambda s: np.nansum(s.astype(float).values)),
        )
        .reset_index()
    )

    inv_grp["Invoices_Pct_Late"] = safe_div(inv_grp["Invoices_Late_Count"], inv_grp["Invoices_Count"])

    # ----------------------------
    # Merge + save
    # ----------------------------

    supplier_mart = (
        po_grp
        .merge(inv_grp, on=["Supplier_ID", "Year"], how="outer")
        .merge(dim_supplier, on="Supplier_ID", how="left")
    )

    # Ensure numeric dtypes for outputs
    numeric_cols = [
        "PO_Lines_Count",
        "PO_Units_Ordered_Year", "PO_Units_Received_Year",
        "PO_LT_Planned_Avg", "PO_LT_Actual_Avg",
        "PO_Late_Count", "PO_FillRate_Year", "PO_Pct_Late",
        "Invoices_Count", "Invoiced_Amount_Year",
        "Payment_Delay_Days_Avg", "Invoices_Late_Count", "Invoices_Pct_Late"
    ]
    for c in numeric_cols:
        if c in supplier_mart.columns:
            supplier_mart[c] = pd.to_numeric(supplier_mart[c], errors="coerce")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "mart_supplier_scorecard.csv"
    supplier_mart.to_csv(out_file, index=False, encoding="utf-8-sig")

    LOG.info("Saved %-28s shape=%s", out_file.name, supplier_mart.shape)
    return supplier_mart


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mart_supplier_scorecard.csv from PO + invoices.")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Folder with raw CSV files.")
    p.add_argument("--output-dir", type=Path, default=Path("data/marts"), help="Folder to write marts.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    build_supplier_scorecard(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
