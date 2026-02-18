from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)


# ----------------------------
# Result container
# ----------------------------

@dataclass
class QAResult:
    issues: list[str]

    def add(self, msg: str) -> None:
        self.issues.append(msg)

    def ok(self) -> bool:
        return len(self.issues) == 0


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

    LOG.info("Loaded %-34s shape=%s", path.name, df.shape)
    return df


def to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def normalize_yearmonth(dim_date: pd.DataFrame) -> pd.DataFrame:
    """
    Build YearMonth=YYYYMM if Year/MonthNumber exist; otherwise keep existing YearMonth.
    """
    dd = dim_date.copy()
    if {"Year", "MonthNumber"}.issubset(dd.columns):
        dd["Year"] = pd.to_numeric(dd["Year"], errors="coerce")
        dd["MonthNumber"] = pd.to_numeric(dd["MonthNumber"], errors="coerce")
        mask = dd["Year"].notna() & dd["MonthNumber"].notna()
        dd.loc[mask, "YearMonth"] = (
            dd.loc[mask, "Year"].astype(int).astype(str)
            + dd.loc[mask, "MonthNumber"].astype(int).astype(str).str.zfill(2)
        )
        dd["YearMonth"] = dd["YearMonth"].astype(str)
    return dd


def orphan_report(
    fact_name: str,
    fact_df: pd.DataFrame,
    key_col: str,
    dim_name: str,
    dim_df: pd.DataFrame,
    dim_key_col: str,
    res: QAResult,
    *,
    allow_nulls: bool = True,
) -> int:
    """
    FK check: keys in fact must exist in dimension (nulls optionally allowed).
    """
    if key_col not in fact_df.columns or dim_key_col not in dim_df.columns:
        LOG.warning("SKIP FK: %s.%s vs %s.%s (missing column)", fact_name, key_col, dim_name, dim_key_col)
        return 0

    keys = fact_df[key_col]
    if allow_nulls:
        keys = keys.dropna()

    fact_keys = pd.Series(keys.unique())
    dim_keys = set(dim_df[dim_key_col].dropna().unique())

    orphans = fact_keys[~fact_keys.isin(dim_keys)]
    n_bad = int(orphans.size)

    if n_bad == 0:
        LOG.info("OK   FK: %s.%s matches %s.%s", fact_name, key_col, dim_name, dim_key_col)
    else:
        LOG.warning("WARN FK: %s.%s has %d orphan keys vs %s", fact_name, key_col, n_bad, dim_name)
        LOG.warning("         examples: %s", list(orphans.head(10)))
        res.add(f"FK orphan keys: {fact_name}.{key_col} -> {dim_name}.{dim_key_col} ({n_bad})")

    return n_bad


def metric_diff_report(
    df: pd.DataFrame,
    col_mart: str,
    col_raw: str,
    label: str,
    res: QAResult,
    *,
    tol: float = 1e-6,
    fail_if_any: bool = True,
) -> None:
    """
    Compare two numeric columns row-wise; report share of diffs > tol.
    """
    if col_mart not in df.columns or col_raw not in df.columns:
        LOG.warning("SKIP DIFF: %s (missing %s/%s)", label, col_mart, col_raw)
        return

    x = pd.to_numeric(df[col_mart], errors="coerce")
    y = pd.to_numeric(df[col_raw], errors="coerce")

    diff = (x - y).abs()
    n_total = int(len(df))
    n_bad = int((diff > tol).sum())
    pct_bad = (n_bad / n_total) if n_total else np.nan

    LOG.info("DIFF %-30s bad=%d share=%.4f (tol=%g)", label, n_bad, pct_bad if not np.isnan(pct_bad) else -1, tol)

    if fail_if_any and n_bad > 0:
        res.add(f"Metric mismatch: {label} ({n_bad} rows > {tol})")


def print_section(title: str) -> None:
    LOG.info("")
    LOG.info("=" * 90)
    LOG.info("%s", title)
    LOG.info("=" * 90)


# ----------------------------
# Core QA
# ----------------------------

def run_qa(input_dir: Path, marts_dir: Path) -> QAResult:
    res = QAResult(issues=[])

    # ---- Stage 0: load
    print_section("STAGE 0 — LOADING RAW TABLES & MARTS")

    dim_date = normalize_yearmonth(read_csv_auto(input_dir / "dim_date.csv"))
    dim_product = read_csv_auto(input_dir / "dim_product.csv")
    dim_store = read_csv_auto(input_dir / "dim_store.csv")
    dim_supplier = read_csv_auto(input_dir / "dim_supplier.csv")
    dim_channel = read_csv_auto(input_dir / "dim_channel.csv")
    dim_currency = read_csv_auto(input_dir / "dim_currency.csv")

    fact_sales = read_csv_auto(input_dir / "fact_sales.csv")
    fact_returns = read_csv_auto(input_dir / "fact_returns.csv")
    fact_inventory_snapshot = read_csv_auto(input_dir / "fact_inventory_snapshot.csv")
    fact_purchase_orders = read_csv_auto(input_dir / "fact_purchase_orders.csv")
    fact_supplier_invoices = read_csv_auto(input_dir / "fact_supplier_invoices.csv")
    fact_logistics_events = read_csv_auto(input_dir / "fact_logistics_events.csv")

    mart_inventory = read_csv_auto(marts_dir / "mart_inventory_monthly.csv")
    mart_bridge = read_csv_auto(marts_dir / "mart_bridge_sales_inventory_po.csv")
    mart_supplier = read_csv_auto(marts_dir / "mart_supplier_scorecard.csv")
    mart_dc_flows = read_csv_auto(marts_dir / "mart_dc_flows.csv")

    # ---- Stage 1: FK checks
    print_section("STAGE 1 — PK/FK INTEGRITY CHECKS")

    LOG.info("[1.1] fact_sales foreign keys")
    orphan_report("fact_sales", fact_sales, "Date_ID", "dim_date", dim_date, "Date_ID", res)
    orphan_report("fact_sales", fact_sales, "Product_ID", "dim_product", dim_product, "Product_ID", res)
    orphan_report("fact_sales", fact_sales, "Channel_ID", "dim_channel", dim_channel, "Channel_ID", res)
    orphan_report("fact_sales", fact_sales, "Store_ID", "dim_store", dim_store, "Store_ID", res)
    orphan_report("fact_sales", fact_sales, "Currency_ID", "dim_currency", dim_currency, "Currency_ID", res)

    LOG.info("[1.2] fact_returns foreign keys")
    orphan_report("fact_returns", fact_returns, "Product_ID", "dim_product", dim_product, "Product_ID", res)
    orphan_report("fact_returns", fact_returns, "Channel_ID", "dim_channel", dim_channel, "Channel_ID", res)
    orphan_report("fact_returns", fact_returns, "Store_ID", "dim_store", dim_store, "Store_ID", res)
    orphan_report("fact_returns", fact_returns, "Currency_ID", "dim_currency", dim_currency, "Currency_ID", res)
    orphan_report("fact_returns", fact_returns, "Date_ID", "dim_date", dim_date, "Date_ID", res)
    orphan_report("fact_returns", fact_returns, "Date_ID_Return", "dim_date", dim_date, "Date_ID", res)

    LOG.info("[1.3] fact_inventory_snapshot foreign keys")
    orphan_report("fact_inventory_snapshot", fact_inventory_snapshot, "Date_ID", "dim_date", dim_date, "Date_ID", res)
    orphan_report("fact_inventory_snapshot", fact_inventory_snapshot, "Product_ID", "dim_product", dim_product, "Product_ID", res)
    orphan_report("fact_inventory_snapshot", fact_inventory_snapshot, "Store_ID", "dim_store", dim_store, "Store_ID", res)

    LOG.info("[1.4] fact_purchase_orders foreign keys")
    orphan_report("fact_purchase_orders", fact_purchase_orders, "Product_ID", "dim_product", dim_product, "Product_ID", res)
    orphan_report("fact_purchase_orders", fact_purchase_orders, "Store_ID", "dim_store", dim_store, "Store_ID", res)
    orphan_report("fact_purchase_orders", fact_purchase_orders, "Supplier_ID", "dim_supplier", dim_supplier, "Supplier_ID", res)
    orphan_report("fact_purchase_orders", fact_purchase_orders, "Currency_ID", "dim_currency", dim_currency, "Currency_ID", res)
    orphan_report("fact_purchase_orders", fact_purchase_orders, "Date_ID_PO", "dim_date", dim_date, "Date_ID", res)
    orphan_report("fact_purchase_orders", fact_purchase_orders, "Date_ID_ExpectedArrival", "dim_date", dim_date, "Date_ID", res)
    orphan_report("fact_purchase_orders", fact_purchase_orders, "Date_ID_ActualArrival", "dim_date", dim_date, "Date_ID", res)

    LOG.info("[1.5] fact_supplier_invoices foreign keys")
    orphan_report("fact_supplier_invoices", fact_supplier_invoices, "Supplier_ID", "dim_supplier", dim_supplier, "Supplier_ID", res)
    orphan_report("fact_supplier_invoices", fact_supplier_invoices, "Currency_ID", "dim_currency", dim_currency, "Currency_ID", res)
    orphan_report("fact_supplier_invoices", fact_supplier_invoices, "Date_ID_Invoice", "dim_date", dim_date, "Date_ID", res)
    orphan_report("fact_supplier_invoices", fact_supplier_invoices, "Date_ID_Due", "dim_date", dim_date, "Date_ID", res)
    orphan_report("fact_supplier_invoices", fact_supplier_invoices, "Date_ID_Paid", "dim_date", dim_date, "Date_ID", res)

    LOG.info("[1.6] fact_supplier_invoices vs fact_purchase_orders (PO_ID + PO_Line_ID)")
    if {"PO_ID", "PO_Line_ID"}.issubset(fact_supplier_invoices.columns) and {"PO_ID", "PO_Line_ID"}.issubset(fact_purchase_orders.columns):
        po_keys = set(zip(fact_purchase_orders["PO_ID"], fact_purchase_orders["PO_Line_ID"]))
        inv_keys = set(zip(fact_supplier_invoices["PO_ID"], fact_supplier_invoices["PO_Line_ID"]))
        missing = [k for k in inv_keys if k not in po_keys]
        if not missing:
            LOG.info("OK   PO linkage: all invoice PO lines exist in fact_purchase_orders")
        else:
            LOG.warning("WARN PO linkage: %d invoice PO lines missing in fact_purchase_orders", len(missing))
            LOG.warning("             examples: %s", missing[:10])
            res.add("Invoices with PO lines not present in fact_purchase_orders")
    else:
        LOG.warning("SKIP PO linkage check (columns missing)")

    LOG.info("[1.7] fact_logistics_events foreign keys")
    # keep raw column names as you used them (lowercase)
    orphan_report("fact_logistics_events", fact_logistics_events, "product_id", "dim_product", dim_product, "Product_ID", res, allow_nulls=True)
    orphan_report("fact_logistics_events", fact_logistics_events, "from_store_id", "dim_store", dim_store, "Store_ID", res, allow_nulls=True)
    orphan_report("fact_logistics_events", fact_logistics_events, "to_store_id", "dim_store", dim_store, "Store_ID", res, allow_nulls=True)
    orphan_report("fact_logistics_events", fact_logistics_events, "supplier_id", "dim_supplier", dim_supplier, "Supplier_ID", res, allow_nulls=True)
    orphan_report("fact_logistics_events", fact_logistics_events, "channel_id", "dim_channel", dim_channel, "Channel_ID", res, allow_nulls=True)
    orphan_report("fact_logistics_events", fact_logistics_events, "date_id_event", "dim_date", dim_date, "Date_ID", res, allow_nulls=True)

    # ---- Stage 2: Inventory mart QA
    print_section("STAGE 2 — INVENTORY MART QA (mart_inventory_monthly vs fact_inventory_snapshot)")

    if {"Date_ID", "Product_ID", "Store_ID"}.issubset(mart_inventory.columns):
        dup = int(mart_inventory.duplicated(subset=["Date_ID", "Product_ID", "Store_ID"]).sum())
        LOG.info("[2.1] mart_inventory_monthly duplicates on (Date_ID, Product_ID, Store_ID): %d", dup)
        if dup > 0:
            res.add("Duplicates in mart_inventory_monthly on PK grain")
    else:
        LOG.warning("[2.1] mart_inventory_monthly missing PK columns")
        res.add("mart_inventory_monthly missing Date_ID/Product_ID/Store_ID")

    LOG.info("[2.2] fact_inventory_snapshot rows: %d", fact_inventory_snapshot.shape[0])
    LOG.info("      mart_inventory_monthly rows: %d", mart_inventory.shape[0])
    if fact_inventory_snapshot.shape[0] != mart_inventory.shape[0]:
        res.add("Row count mismatch: fact_inventory_snapshot vs mart_inventory_monthly")

    inv = mart_inventory.copy()
    inv_cols = ["Stock_Begin", "Units_Sold", "Units_Replenished", "Units_Shrinkage", "Units_Returned_Resellable", "Stock_End"]
    missing_inv = [c for c in inv_cols if c not in inv.columns]
    if missing_inv:
        LOG.warning("[2.3] SKIP inventory equation check (missing columns: %s)", missing_inv)
    else:
        to_numeric(inv, inv_cols)
        inv["Stock_Theoretical"] = (
            inv["Stock_Begin"]
            - inv["Units_Sold"]
            + inv["Units_Replenished"]
            + inv["Units_Returned_Resellable"]
            - inv["Units_Shrinkage"]
        )
        inv["Stock_Diff"] = inv["Stock_Theoretical"] - inv["Stock_End"]
        total = int(len(inv))
        non_zero = int((inv["Stock_Diff"].round(6) != 0).sum())
        pct_non_zero = non_zero / total if total else np.nan

        LOG.info("[2.3] Inventory equation diff!=0 rows: %d (share=%.4f)", non_zero, pct_non_zero if not np.isnan(pct_non_zero) else -1)

        # tolerance policy: allow <=3% as modeled noise
        if not np.isnan(pct_non_zero) and pct_non_zero > 0.03:
            res.add("Inventory equation has >3% mismatching rows")

    # ---- Stage 3: Bridge mart QA (rebuild raw aggregates and compare)
    print_section("STAGE 3 — BRIDGE MART QA (mart_bridge_sales_inventory_po)")

    if "YearMonth" not in dim_date.columns:
        raise ValueError("dim_date must contain YearMonth (or Year+MonthNumber) for bridge reconciliation.")
    dim_date_ym = dim_date[["Date_ID", "YearMonth"]].drop_duplicates()

    key_cols = ["Product_ID", "Store_ID", "YearMonth"]

    # raw sales agg
    fs = fact_sales.merge(dim_date_ym, on="Date_ID", how="left")
    to_numeric(fs, ["Units_Sold", "Net_Sales", "Gross_Profit"])
    raw_sales_m = (
        fs.groupby(key_cols, dropna=False)
        .agg(
            Units_Sold_Month=("Units_Sold", "sum"),
            Net_Sales_Month=("Net_Sales", "sum"),
            Gross_Profit_Month=("Gross_Profit", "sum"),
        )
        .reset_index()
    )

    # raw returns agg (Date_ID_Return -> YearMonth)
    fr = fact_returns.merge(dim_date_ym.rename(columns={"Date_ID": "Date_ID_Return"}), on="Date_ID_Return", how="left")
    to_numeric(fr, ["Return_Qty"])
    raw_returns_m = (
        fr.groupby(key_cols, dropna=False)
        .agg(Return_Qty_Month=("Return_Qty", "sum"))
        .reset_index()
    )

    # raw inventory agg
    fi = fact_inventory_snapshot.merge(dim_date_ym, on="Date_ID", how="left")
    to_numeric(fi, ["Units_Replenished", "Units_Shrinkage", "Stock_End"])
    raw_inv_m = (
        fi.groupby(key_cols, dropna=False)
        .agg(
            Units_Replenished_Month=("Units_Replenished", "sum"),
            Units_Shrinkage_Month=("Units_Shrinkage", "sum"),
            Stock_End_Month=("Stock_End", "max"),
        )
        .reset_index()
    )

    # raw PO agg (ActualArrival -> YearMonth)
    fpo = fact_purchase_orders.merge(
        dim_date_ym.rename(columns={"Date_ID": "Date_ID_ActualArrival"}),
        on="Date_ID_ActualArrival",
        how="left",
    )
    to_numeric(fpo, ["Order_Qty", "Received_Qty", "Lead_Time_Days_Actual"])
    raw_po_m = (
        fpo.groupby(key_cols, dropna=False)
        .agg(
            PO_Qty_Ordered_Month=("Order_Qty", "sum"),
            PO_Qty_Received_Month=("Received_Qty", "sum"),
            PO_LT_Actual_Avg=("Lead_Time_Days_Actual", "mean"),
        )
        .reset_index()
    )

    raw_bridge = (
        raw_sales_m
        .merge(raw_returns_m, how="left", on=key_cols)
        .merge(raw_inv_m, how="left", on=key_cols)
        .merge(raw_po_m, how="left", on=key_cols)
    )

    # duplicates check (mart + raw)
    if set(key_cols).issubset(mart_bridge.columns):
        dup_m = int(mart_bridge.duplicated(subset=key_cols).sum())
        LOG.info("[3.4] Duplicates in mart_bridge on grain: %d", dup_m)
        if dup_m > 0:
            res.add("Duplicates in mart_bridge on grain Product_ID/Store_ID/YearMonth")
    else:
        res.add("mart_bridge missing key columns for reconciliation")

    dup_r = int(raw_bridge.duplicated(subset=key_cols).sum())
    LOG.info("[3.4] Duplicates in raw_bridge on grain: %d", dup_r)
    if dup_r > 0:
        res.add("Duplicates in raw_bridge on grain Product_ID/Store_ID/YearMonth")

    comp = mart_bridge.merge(raw_bridge, how="outer", on=key_cols, suffixes=("_mart", "_raw"), indicator=True)
    LOG.info("[3.4] Merge result counts: %s", comp["_merge"].value_counts().to_dict())

    metric_names = [
        "Units_Sold_Month",
        "Net_Sales_Month",
        "Gross_Profit_Month",
        "Return_Qty_Month",
        "Units_Replenished_Month",
        "Units_Shrinkage_Month",
        "Stock_End_Month",
        "PO_Qty_Ordered_Month",
        "PO_Qty_Received_Month",
    ]
    print_section("STAGE 3.5 — BRIDGE METRIC DIFFS (mart vs raw)")

    for m in metric_names:
        metric_diff_report(comp, f"{m}_mart", f"{m}_raw", f"bridge:{m}", res, tol=1e-6, fail_if_any=True)

    # ---- Stage 4: Supplier scorecard QA (re-aggregate raw and compare)
    print_section("STAGE 4 — SUPPLIER SCORECARD QA (mart_supplier_scorecard)")

    sup = mart_supplier.copy()
    to_numeric(sup, [
        "PO_Lines_Count", "PO_Units_Ordered_Year", "PO_Units_Received_Year",
        "PO_LT_Planned_Avg", "PO_LT_Actual_Avg", "PO_Late_Count", "PO_FillRate_Year",
        "PO_Pct_Late",
        "Invoices_Count", "Invoiced_Amount_Year",
        "Payment_Delay_Days_Avg", "Invoices_Late_Count", "Invoices_Pct_Late"
    ])

    po = fact_purchase_orders.merge(
        dim_date[["Date_ID", "Year"]],
        left_on="Date_ID_PO",
        right_on="Date_ID",
        how="left"
    )
    to_numeric(po, ["Order_Qty", "Received_Qty", "Lead_Time_Days_Planned", "Lead_Time_Days_Actual"])

    # NOTE: keep your original column name if present
    late_col = "Is_Late_PO" if "Is_Late_PO" in po.columns else ("Is_Late" if "Is_Late" in po.columns else None)
    if late_col:
        s = po[late_col].astype(str).str.strip().str.lower().isin(["1", "true", "y", "yes"])
        po["Is_Late_Bin"] = s.astype(int)
    else:
        po["Is_Late_Bin"] = np.nan

    po_agg = (
        po.groupby(["Supplier_ID", "Year"], dropna=False)
        .agg(
            PO_Lines_Count_raw=("PO_Line_ID", "count"),
            PO_Units_Ordered_Year_raw=("Order_Qty", "sum"),
            PO_Units_Received_Year_raw=("Received_Qty", "sum"),
            PO_LT_Planned_Avg_raw=("Lead_Time_Days_Planned", "mean"),
            PO_LT_Actual_Avg_raw=("Lead_Time_Days_Actual", "mean"),
            PO_Late_Count_raw=("Is_Late_Bin", "sum"),
        )
        .reset_index()
    )

    inv = fact_supplier_invoices.merge(
        dim_date[["Date_ID", "Year"]],
        left_on="Date_ID_Invoice",
        right_on="Date_ID",
        how="left"
    )
    to_numeric(inv, ["Invoiced_Amount", "Payment_Delay_Days"])

    inv_late_col = "Is_Invoice_Late" if "Is_Invoice_Late" in inv.columns else None
    if inv_late_col:
        s = inv[inv_late_col].astype(str).str.strip().str.lower().isin(["1", "true", "y", "yes"])
        inv["Is_Invoice_Late_Bin"] = s.astype(int)
    else:
        inv["Is_Invoice_Late_Bin"] = np.nan

    inv_agg = (
        inv.groupby(["Supplier_ID", "Year"], dropna=False)
        .agg(
            Invoices_Count_raw=("Invoice_ID", "nunique"),
            Invoiced_Amount_Year_raw=("Invoiced_Amount", "sum"),
            Payment_Delay_Days_Avg_raw=("Payment_Delay_Days", "mean"),
            Invoices_Late_Count_raw=("Is_Invoice_Late_Bin", "sum"),
        )
        .reset_index()
    )

    # align Year type
    if "Year" in sup.columns:
        sup["Year"] = sup["Year"].astype(str)
    po_agg["Year"] = po_agg["Year"].astype(str)
    inv_agg["Year"] = inv_agg["Year"].astype(str)

    sup_q = (
        sup.merge(po_agg, on=["Supplier_ID", "Year"], how="left")
           .merge(inv_agg, on=["Supplier_ID", "Year"], how="left")
    )

    metric_diff_report(sup_q, "PO_Lines_Count", "PO_Lines_Count_raw", "supplier:PO_Lines_Count", res)
    metric_diff_report(sup_q, "PO_Units_Ordered_Year", "PO_Units_Ordered_Year_raw", "supplier:PO_Units_Ordered_Year", res)
    metric_diff_report(sup_q, "PO_Units_Received_Year", "PO_Units_Received_Year_raw", "supplier:PO_Units_Received_Year", res)
    metric_diff_report(sup_q, "PO_LT_Planned_Avg", "PO_LT_Planned_Avg_raw", "supplier:PO_LT_Planned_Avg", res)
    metric_diff_report(sup_q, "PO_LT_Actual_Avg", "PO_LT_Actual_Avg_raw", "supplier:PO_LT_Actual_Avg", res)
    metric_diff_report(sup_q, "PO_Late_Count", "PO_Late_Count_raw", "supplier:PO_Late_Count", res)

    metric_diff_report(sup_q, "Invoices_Count", "Invoices_Count_raw", "supplier:Invoices_Count", res)
    metric_diff_report(sup_q, "Invoiced_Amount_Year", "Invoiced_Amount_Year_raw", "supplier:Invoiced_Amount_Year", res)
    metric_diff_report(sup_q, "Payment_Delay_Days_Avg", "Payment_Delay_Days_Avg_raw", "supplier:Payment_Delay_Days_Avg", res)
    metric_diff_report(sup_q, "Invoices_Late_Count", "Invoices_Late_Count_raw", "supplier:Invoices_Late_Count", res)

    # ---- Stage 5: DC flows QA (raw agg -> compare)
    print_section("STAGE 5 — DC FLOWS QA (mart_dc_flows vs fact_logistics_events)")

    dim_date_small = dim_date[["Date_ID", "YearMonth"]].copy()
    dim_store_small = dim_store[["Store_ID", "Is_Store", "Is_DC"]].copy()

    fle = fact_logistics_events.copy()
    fle = fle.merge(dim_date_small, left_on="date_id_event", right_on="Date_ID", how="left")

    fle = fle.merge(
        dim_store_small.rename(columns={"Store_ID": "from_store_id", "Is_Store": "From_Is_Store", "Is_DC": "From_Is_DC"}),
        on="from_store_id",
        how="left",
    )
    fle = fle.merge(
        dim_store_small.rename(columns={"Store_ID": "to_store_id", "Is_Store": "To_Is_Store", "Is_DC": "To_Is_DC"}),
        on="to_store_id",
        how="left",
    )

    to_numeric(fle, ["units_moved", "units_damaged", "transit_days_actual"])
    for c in ["From_Is_Store", "From_Is_DC", "To_Is_Store", "To_Is_DC"]:
        if c in fle.columns:
            fle[c] = pd.to_numeric(fle[c], errors="coerce")

    if "is_late" in fle.columns:
        fle["Late_Flag_Bin"] = fle["is_late"].astype(str).str.strip().str.lower().isin(["1", "true", "y", "yes"]).astype(int)
    else:
        fle["Late_Flag_Bin"] = np.nan

    # vectorized flow type (avoid apply)
    from_id_na = fle["from_store_id"].isna() if "from_store_id" in fle.columns else pd.Series(True, index=fle.index)
    to_id_ok = fle["to_store_id"].notna() if "to_store_id" in fle.columns else pd.Series(False, index=fle.index)

    cond_supplier_to_dc = from_id_na & to_id_ok & (fle["To_Is_DC"] == 1)
    cond_dc_to_store = (fle["From_Is_DC"] == 1) & (fle["To_Is_Store"] == 1)
    cond_store_to_dc = (fle["From_Is_Store"] == 1) & (fle["To_Is_DC"] == 1)
    cond_store_to_store = (fle["From_Is_Store"] == 1) & (fle["To_Is_Store"] == 1)

    fle["Flow_Type_QA"] = np.select(
        [cond_supplier_to_dc, cond_dc_to_store, cond_store_to_dc, cond_store_to_store],
        ["Supplier_to_DC", "DC_to_Store", "Store_to_DC", "Store_to_Store"],
        default="Other",
    )

    # raw agg
    needed_cols = ["YearMonth", "from_store_id", "to_store_id", "Flow_Type_QA"]
    for c in needed_cols:
        if c not in fle.columns:
            raise ValueError(f"fact_logistics_events missing required column for DC QA: {c}")

    fle_agg = (
        fle.groupby(needed_cols, dropna=False)
        .agg(
            Events_Count_raw=("event_id", "count") if "event_id" in fle.columns else (fle.columns[0], "count"),
            Units_Moved_Total_raw=("units_moved", "sum"),
            Damage_Loss_Total_raw=("units_damaged", "sum") if "units_damaged" in fle.columns else ("units_moved", "sum"),
            Transit_Days_Avg_raw=("transit_days_actual", "mean") if "transit_days_actual" in fle.columns else ("units_moved", "mean"),
            Late_Share_raw=("Late_Flag_Bin", "mean"),
        )
        .reset_index()
    )

    dc = mart_dc_flows.copy()
    to_numeric(dc, ["Events_Count", "Units_Moved_Total", "Damage_Loss_Total", "Transit_Days_Avg", "Late_Share"])

    dc_q = dc.merge(
        fle_agg,
        left_on=["YearMonth", "from_store_id", "to_store_id", "Flow_Type"],
        right_on=["YearMonth", "from_store_id", "to_store_id", "Flow_Type_QA"],
        how="left",
    )

    metric_diff_report(dc_q, "Events_Count", "Events_Count_raw", "dc:Events_Count", res)
    metric_diff_report(dc_q, "Units_Moved_Total", "Units_Moved_Total_raw", "dc:Units_Moved_Total", res)
    if "Damage_Loss_Total" in dc_q.columns and "Damage_Loss_Total_raw" in dc_q.columns:
        metric_diff_report(dc_q, "Damage_Loss_Total", "Damage_Loss_Total_raw", "dc:Damage_Loss_Total", res)
    if "Transit_Days_Avg" in dc_q.columns and "Transit_Days_Avg_raw" in dc_q.columns:
        metric_diff_report(dc_q, "Transit_Days_Avg", "Transit_Days_Avg_raw", "dc:Transit_Days_Avg", res)
    metric_diff_report(dc_q, "Late_Share", "Late_Share_raw", "dc:Late_Share", res)

    # ---- Final
    print_section("FINAL QA STATUS")
    if res.ok():
        LOG.info("QA STATUS: PASSED (no critical mismatches detected)")
    else:
        LOG.error("QA STATUS: FAILED (%d issues)", len(res.issues))
        for it in res.issues:
            LOG.error(" - %s", it)

    return res


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QA integrity checks: PK/FK + reconciliation raw vs marts.")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Folder with raw CSV files.")
    p.add_argument("--marts-dir", type=Path, default=Path("data/marts"), help="Folder with mart CSV files.")
    p.add_argument("--fail-on-warn", action="store_true", help="Treat warnings as failures (exit code 1).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    res = run_qa(args.input_dir, args.marts_dir)

    # exit code: fail if issues exist
    if args.fail_on_warn and not res.ok():
        return 1
    if not res.ok():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
