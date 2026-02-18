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

def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        dtype=str,
    )
    LOG.info("Loaded %-32s shape=%s", path.name, df.shape)
    return df


def to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom2 = denom.replace({0: np.nan})
    return numer / denom2


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    LOG.info("Saved  %-32s shape=%s", path.name, df.shape)


# ----------------------------
# Core
# ----------------------------

def run_diagnostics(bridge_path: Path, output_dir: Path) -> None:
    df = read_csv_auto(bridge_path)

    required = ["Product_ID", "Units_Sold_Month", "Units_Replenished_Month", "Category", "Store_Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bridge mart missing required columns: {missing}")

    num_cols = [
        "Units_Sold_Month",
        "Net_Sales_Month",
        "Gross_Profit_Month",
        "Return_Qty_Month",
        "Units_Replenished_Month",
        "Units_Shrinkage_Month",
        "Stock_End_Month",
        "PO_Qty_Ordered_Month",
        "PO_Qty_Received_Month",
        "PO_LT_Actual_Avg",
    ]
    to_numeric(df, num_cols)

    df["Demand_Supply_Gap"] = df["Units_Replenished_Month"] - df["Units_Sold_Month"]
    df["Repl_Sales_Ratio"] = safe_div(df["Units_Replenished_Month"], df["Units_Sold_Month"])

    df["Overbuy_Flag"] = df["Demand_Supply_Gap"] > 0
    df["Underbuy_Flag"] = df["Demand_Supply_Gap"] < 0

    # Category level
    cat = (
        df.groupby("Category", dropna=False)
        .agg(
            Rows=("Product_ID", "size"),
            Avg_Gap=("Demand_Supply_Gap", "mean"),
            Overbuy_Pct=("Overbuy_Flag", "mean"),
            Underbuy_Pct=("Underbuy_Flag", "mean"),
            Ratio_Avg=("Repl_Sales_Ratio", "mean"),
        )
        .reset_index()
        .sort_values(["Rows"], ascending=False)
    )
    write_csv(cat, output_dir / "demand_supply_by_category.csv")

    # Store type level
    st = (
        df.groupby("Store_Type", dropna=False)
        .agg(
            Rows=("Product_ID", "size"),
            Avg_Gap=("Demand_Supply_Gap", "mean"),
            Overbuy_Pct=("Overbuy_Flag", "mean"),
            Underbuy_Pct=("Underbuy_Flag", "mean"),
            Ratio_Avg=("Repl_Sales_Ratio", "mean"),
        )
        .reset_index()
        .sort_values(["Rows"], ascending=False)
    )
    write_csv(st, output_dir / "demand_supply_by_store_type.csv")

    # ABC/XYZ (optional)
    if "ABC_Class" in df.columns and "XYZ_Class" in df.columns:
        abcxyz = (
            df.groupby(["ABC_Class", "XYZ_Class"], dropna=False)
            .agg(
                Rows=("Product_ID", "size"),
                Avg_Gap=("Demand_Supply_Gap", "mean"),
                Overbuy_Pct=("Overbuy_Flag", "mean"),
                Underbuy_Pct=("Underbuy_Flag", "mean"),
                Ratio_Avg=("Repl_Sales_Ratio", "mean"),
            )
            .reset_index()
            .sort_values(["Rows"], ascending=False)
        )
        write_csv(abcxyz, output_dir / "demand_supply_abcxyz.csv")
    else:
        LOG.info("ABC/XYZ columns not found — skipping abcxyz output.")

    # Heatmap Category × Store_Type
    heat = (
        df.groupby(["Category", "Store_Type"], dropna=False)
        .agg(
            Avg_Gap=("Demand_Supply_Gap", "mean"),
            Overbuy_Pct=("Overbuy_Flag", "mean"),
            Underbuy_Pct=("Underbuy_Flag", "mean"),
        )
        .reset_index()
    )
    write_csv(heat, output_dir / "demand_supply_heatmap.csv")

    LOG.info("Diagnostics completed.")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demand–Supply diagnostics from mart_bridge_sales_inventory_po.csv")
    p.add_argument(
        "--bridge-path",
        type=Path,
        default=Path("data/marts/mart_bridge_sales_inventory_po.csv"),
        help="Path to bridge mart CSV.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics"),
        help="Folder to write diagnostic outputs.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    run_diagnostics(args.bridge_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
