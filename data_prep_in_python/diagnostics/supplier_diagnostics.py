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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    LOG.info("Saved  %-32s shape=%s", path.name, df.shape)


def summarize_series(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return pd.Series(
            {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "max": np.nan}
        )
    return pd.Series(
        {
            "count": s.count(),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": s.quantile(0.25),
            "p50": s.median(),
            "p75": s.quantile(0.75),
            "max": s.max(),
        }
    )


def ratio(series: pd.Series) -> float:
    # returns share of True among non-null values
    if series is None:
        return np.nan
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


# ----------------------------
# Core
# ----------------------------

def run(sc_path: Path, output_dir: Path) -> None:
    sc = read_csv_auto(sc_path)

    numeric_cols = [
        "PO_Lines_Count",
        "PO_Units_Ordered_Year", "PO_Units_Received_Year",
        "PO_LT_Planned_Avg", "PO_LT_Actual_Avg",
        "PO_Late_Count", "PO_FillRate_Year", "PO_Pct_Late",
        "Invoices_Count", "Invoiced_Amount_Year",
        "Payment_Delay_Days_Avg", "Invoices_Late_Count", "Invoices_Pct_Late",
        "Reliability_Score",
    ]
    to_numeric(sc, numeric_cols)

    # A) Overall summary for core metrics
    metrics = [
        "PO_FillRate_Year",
        "PO_LT_Actual_Avg",
        "PO_LT_Planned_Avg",
        "PO_Pct_Late",
        "Invoices_Pct_Late",
        "Payment_Delay_Days_Avg",
    ]

    rows = []
    for col in metrics:
        if col in sc.columns:
            st = summarize_series(sc[col])
            st.name = col
            rows.append(st)

    overall = pd.DataFrame(rows).reset_index().rename(columns={"index": "Metric"})
    write_csv(overall, output_dir / "supplier_overall_summary.csv")

    # Extra KPI shares
    extra = []
    if "PO_FillRate_Year" in sc.columns:
        extra.append(("FillRate_lt_0.95_Pct", ratio(sc["PO_FillRate_Year"] < 0.95)))
        extra.append(("FillRate_gt_1.02_Pct", ratio(sc["PO_FillRate_Year"] > 1.02)))
    if "PO_Pct_Late" in sc.columns:
        extra.append(("PO_Pct_Late_gt_0.10_Share", ratio(sc["PO_Pct_Late"] > 0.10)))
    if "Invoices_Pct_Late" in sc.columns:
        extra.append(("Invoices_Pct_Late_gt_0.10_Share", ratio(sc["Invoices_Pct_Late"] > 0.10)))

    extra_df = pd.DataFrame(extra, columns=["Metric", "Value"])
    write_csv(extra_df, output_dir / "supplier_overall_extra_kpi.csv")

    # B) By region
    if "Region" in sc.columns:
        by_region = (
            sc.groupby("Region", dropna=False)
            .agg(
                Suppliers=("Supplier_ID", "nunique"),
                Years=("Year", "nunique"),
                PO_Lines=("PO_Lines_Count", "sum"),
                Avg_FillRate=("PO_FillRate_Year", "mean"),
                Avg_LT_Planned=("PO_LT_Planned_Avg", "mean"),
                Avg_LT_Actual=("PO_LT_Actual_Avg", "mean"),
                Avg_PO_Pct_Late=("PO_Pct_Late", "mean"),
                Avg_Invoices_Pct_Late=("Invoices_Pct_Late", "mean"),
                Avg_Payment_Delay_Days=("Payment_Delay_Days_Avg", "mean"),
            )
            .reset_index()
            .sort_values("Suppliers", ascending=False)
        )
        write_csv(by_region, output_dir / "supplier_by_region.csv")
    else:
        LOG.info("Region not found — skipping supplier_by_region.csv")

    # C) By supplier category
    if "Supplier_Category" in sc.columns:
        by_cat = (
            sc.groupby("Supplier_Category", dropna=False)
            .agg(
                Suppliers=("Supplier_ID", "nunique"),
                Years=("Year", "nunique"),
                PO_Lines=("PO_Lines_Count", "sum"),
                Avg_FillRate=("PO_FillRate_Year", "mean"),
                Avg_LT_Planned=("PO_LT_Planned_Avg", "mean"),
                Avg_LT_Actual=("PO_LT_Actual_Avg", "mean"),
                Avg_PO_Pct_Late=("PO_Pct_Late", "mean"),
                Avg_Invoices_Pct_Late=("Invoices_Pct_Late", "mean"),
                Avg_Payment_Delay_Days=("Payment_Delay_Days_Avg", "mean"),
            )
            .reset_index()
            .sort_values("Suppliers", ascending=False)
        )
        write_csv(by_cat, output_dir / "supplier_by_category.csv")
    else:
        LOG.info("Supplier_Category not found — skipping supplier_by_category.csv")

    # D) By reliability band
    if "Reliability_Score" in sc.columns:
        # Bands: <0.80 / 0.80–0.90 / >0.90 / Unknown
        bins = [-np.inf, 0.80, 0.90, np.inf]
        labels = ["<0.80", "0.80–0.90", ">0.90"]
        sc["Reliability_Band"] = pd.cut(sc["Reliability_Score"], bins=bins, labels=labels, right=False)
        sc.loc[sc["Reliability_Score"].isna(), "Reliability_Band"] = "Unknown"

        by_rel = (
            sc.groupby("Reliability_Band", dropna=False)
            .agg(
                Suppliers=("Supplier_ID", "nunique"),
                Years=("Year", "nunique"),
                PO_Lines=("PO_Lines_Count", "sum"),
                Avg_FillRate=("PO_FillRate_Year", "mean"),
                Avg_LT_Planned=("PO_LT_Planned_Avg", "mean"),
                Avg_LT_Actual=("PO_LT_Actual_Avg", "mean"),
                Avg_PO_Pct_Late=("PO_Pct_Late", "mean"),
                Avg_Invoices_Pct_Late=("Invoices_Pct_Late", "mean"),
                Avg_Payment_Delay_Days=("Payment_Delay_Days_Avg", "mean"),
            )
            .reset_index()
        )
        write_csv(by_rel, output_dir / "supplier_by_reliability_band.csv")
    else:
        LOG.info("Reliability_Score not found — skipping supplier_by_reliability_band.csv")

    LOG.info("Supplier scorecard diagnostics completed.")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostics for mart_supplier_scorecard.csv")
    p.add_argument(
        "--scorecard-path",
        type=Path,
        default=Path("data/marts/mart_supplier_scorecard.csv"),
        help="Path to mart_supplier_scorecard.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics"),
        help="Folder to write diagnostic outputs",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    run(args.scorecard_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
