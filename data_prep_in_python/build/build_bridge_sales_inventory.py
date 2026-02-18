from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOG = logging.getLogger(__name__)


# ----------------------------
# I/O helpers
# ----------------------------

def read_csv_auto(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep=None,              # auto-detect delimiter
        engine="python",
        encoding="utf-8-sig",
        dtype=str
    )

    if usecols is not None:
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns: {missing}")
        df = df[usecols].copy()

    LOG.info("Loaded %-30s shape=%s", path.name, df.shape)
    return df


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Transform steps
# ----------------------------

def build_yearmonth(dim_date: pd.DataFrame) -> pd.DataFrame:
    required = ["Date_ID", "Year", "MonthNumber"]
    missing = [c for c in required if c not in dim_date.columns]
    if missing:
        raise ValueError(f"dim_date: missing columns: {missing}")

    dd = dim_date[required].copy()
    dd["Year"] = pd.to_numeric(dd["Year"], errors="coerce")
    dd["MonthNumber"] = pd.to_numeric(dd["MonthNumber"], errors="coerce")

    mask = dd["Year"].notna() & dd["MonthNumber"].notna()
    dd.loc[mask, "YearMonth"] = (
        dd.loc[mask, "Year"].astype(int).astype(str)
        + dd.loc[mask, "MonthNumber"].astype(int).astype(str).str.zfill(2)
    )

    out = dd[["Date_ID", "YearMonth"]].copy()
    out["YearMonth"] = out["YearMonth"].astype(str)
    return out


def agg_sales_monthly(fact_sales: pd.DataFrame) -> pd.DataFrame:
    fact_sales = to_numeric(fact_sales, ["Units_Sold", "Net_Sales", "Gross_Profit"])
    return (
        fact_sales
        .groupby(["Product_ID", "Store_ID", "YearMonth"], dropna=False)
        .agg(
            Units_Sold_Month=("Units_Sold", "sum"),
            Net_Sales_Month=("Net_Sales", "sum"),
            Gross_Profit_Month=("Gross_Profit", "sum"),
        )
        .reset_index()
    )


def agg_returns_monthly(fact_returns: pd.DataFrame) -> pd.DataFrame:
    fact_returns = to_numeric(fact_returns, ["Return_Qty"])
    return (
        fact_returns
        .groupby(["Product_ID", "Store_ID", "YearMonth"], dropna=False)
        .agg(Return_Qty_Month=("Return_Qty", "sum"))
        .reset_index()
    )


def agg_inventory_monthly(fact_inventory: pd.DataFrame) -> pd.DataFrame:
    fact_inventory = to_numeric(fact_inventory, ["Units_Replenished", "Units_Shrinkage", "Stock_End"])

    # Inventory snapshot is EOM-grain; Stock_End should be taken as max (one row expected)
    return (
        fact_inventory
        .groupby(["Product_ID", "Store_ID", "YearMonth"], dropna=False)
        .agg(
            Units_Replenished_Month=("Units_Replenished", "sum"),
            Units_Shrinkage_Month=("Units_Shrinkage", "sum"),
            Stock_End_Month=("Stock_End", "max"),
        )
        .reset_index()
    )


def agg_po_monthly(fact_po: pd.DataFrame) -> pd.DataFrame:
    fact_po = to_numeric(fact_po, ["Order_Qty", "Received_Qty", "Lead_Time_Days_Actual"])
    return (
        fact_po
        .groupby(["Product_ID", "Store_ID", "YearMonth"], dropna=False)
        .agg(
            PO_Qty_Ordered_Month=("Order_Qty", "sum"),
            PO_Qty_Received_Month=("Received_Qty", "sum"),
            PO_LT_Actual_Avg=("Lead_Time_Days_Actual", "mean"),
        )
        .reset_index()
    )


def build_bridge(
    dim_date_ym: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_store: pd.DataFrame,
    fact_sales: pd.DataFrame,
    fact_returns: pd.DataFrame,
    fact_inventory: pd.DataFrame,
    fact_po: pd.DataFrame,
) -> pd.DataFrame:

    fact_sales = fact_sales.merge(dim_date_ym, on="Date_ID", how="left")

    fact_returns = fact_returns.merge(
        dim_date_ym.rename(columns={"Date_ID": "Date_ID_Return"}),
        on="Date_ID_Return",
        how="left",
    )

    fact_inventory = fact_inventory.merge(dim_date_ym, on="Date_ID", how="left")

    fact_po = fact_po.merge(
        dim_date_ym.rename(columns={"Date_ID": "Date_ID_ActualArrival"}),
        on="Date_ID_ActualArrival",
        how="left",
    )

    sales_m = agg_sales_monthly(fact_sales)
    returns_m = agg_returns_monthly(fact_returns)
    inv_m = agg_inventory_monthly(fact_inventory)
    po_m = agg_po_monthly(fact_po)

    bridge = (
        sales_m
        .merge(returns_m, how="left", on=["Product_ID", "Store_ID", "YearMonth"])
        .merge(inv_m, how="left", on=["Product_ID", "Store_ID", "YearMonth"])
        .merge(po_m, how="left", on=["Product_ID", "Store_ID", "YearMonth"])
        .merge(dim_product, how="left", on="Product_ID")
        .merge(dim_store, how="left", on="Store_ID")
    )

    numeric_cols = [
        "Units_Sold_Month", "Net_Sales_Month", "Gross_Profit_Month",
        "Return_Qty_Month",
        "Units_Replenished_Month", "Units_Shrinkage_Month", "Stock_End_Month",
        "PO_Qty_Ordered_Month", "PO_Qty_Received_Month", "PO_LT_Actual_Avg",
    ]
    bridge = to_numeric(bridge, numeric_cols)

    return bridge


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mart_bridge_sales_inventory_po.csv from raw tables.")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Folder with raw CSV files.")
    p.add_argument("--output-dir", type=Path, default=Path("data/marts"), help="Folder to write marts.")
    p.add_argument("--output-name", type=str, default="mart_bridge_sales_inventory_po.csv", help="Output CSV file name.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    inp = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    dim_date_raw = read_csv_auto(inp / "dim_date.csv", usecols=["Date_ID", "Year", "MonthNumber"])
    dim_date_ym = build_yearmonth(dim_date_raw)

    dim_product = read_csv_auto(
        inp / "dim_product.csv",
        usecols=["Product_ID", "Brand", "Category", "Subcategory", "Price_Band", "Lifecycle_Flag"]
    )
    dim_store = read_csv_auto(
        inp / "dim_store.csv",
        usecols=["Store_ID", "Store_Type", "Country", "Region"]
    )

    fact_sales = read_csv_auto(
        inp / "fact_sales.csv",
        usecols=["Date_ID", "Product_ID", "Store_ID", "Units_Sold", "Net_Sales", "Gross_Profit"]
    )
    fact_returns = read_csv_auto(
        inp / "fact_returns.csv",
        usecols=["Date_ID_Return", "Product_ID", "Store_ID", "Return_Qty"]
    )
    fact_inventory = read_csv_auto(
        inp / "fact_inventory_snapshot.csv",
        usecols=["Date_ID", "Product_ID", "Store_ID", "Units_Replenished", "Units_Shrinkage", "Stock_End"]
    )
    fact_po = read_csv_auto(
        inp / "fact_purchase_orders.csv",
        usecols=["Date_ID_ActualArrival", "Product_ID", "Store_ID", "Order_Qty", "Received_Qty", "Lead_Time_Days_Actual"]
    )

    bridge = build_bridge(
        dim_date_ym=dim_date_ym,
        dim_product=dim_product,
        dim_store=dim_store,
        fact_sales=fact_sales,
        fact_returns=fact_returns,
        fact_inventory=fact_inventory,
        fact_po=fact_po,
    )

    out_file = out / args.output_name
    bridge.to_csv(out_file, index=False, encoding="utf-8-sig")

    LOG.info("Saved %-30s shape=%s", out_file.name, bridge.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
