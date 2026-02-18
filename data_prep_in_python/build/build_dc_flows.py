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


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first existing column name from candidates (case-insensitive).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def to_numeric(df: pd.DataFrame, col: str | None) -> None:
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def normalize_bool01(series: pd.Series) -> pd.Series:
    """
    Convert common truthy strings to 1/0, keep NaN if unknown.
    """
    s = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "y", "yes"}
    falsy = {"0", "false", "n", "no"}
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s.isin(truthy)] = 1.0
    out[s.isin(falsy)] = 0.0
    return out


# ----------------------------
# Core logic
# ----------------------------

def build_dc_flows(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    # Required dims
    dim_date = read_csv_auto(input_dir / "dim_date.csv", usecols=["Date_ID", "YearMonth"])
    dim_store = read_csv_auto(
        input_dir / "dim_store.csv",
        usecols=["Store_ID", "Store_Type", "Country", "Region", "Is_Store", "Is_DC"],
    )

    fle = read_csv_auto(input_dir / "fact_logistics_events.csv")

    # Column detection (robust against naming variants)
    date_col = pick_column(fle, ["date_id_event", "Date_ID_Event"])
    from_store_col = pick_column(fle, ["from_store_id", "From_Store_ID"])
    to_store_col = pick_column(fle, ["to_store_id", "To_Store_ID"])

    units_actual_col = pick_column(
        fle,
        ["Units_Actual", "units_actual", "Units_Moved", "units_moved", "Qty", "qty", "Quantity", "quantity"],
    )
    units_planned_col = pick_column(
        fle,
        ["Units_Planned", "units_planned", "Planned_Units", "planned_units"],
    )
    damage_col = pick_column(
        fle,
        ["Damage_Qty", "damage_qty", "Units_Damaged", "units_damaged", "Lost_Units", "units_lost", "lost_units"],
    )
    transit_actual_col = pick_column(fle, ["transit_days_actual", "Transit_Days_Actual"])
    transit_planned_col = pick_column(fle, ["transit_days_planned", "Transit_Days_Planned"])
    transit_col = transit_actual_col or transit_planned_col

    late_flag_col = pick_column(fle, ["Is_Late", "is_late", "Late_Flag", "late_flag", "OTIF_Flag", "otif_flag"])
    event_id_col = pick_column(fle, ["event_id", "Event_ID", "Logistics_Event_ID"]) or fle.columns[0]

    LOG.info(
        "Detected columns: date=%s from=%s to=%s units_actual=%s units_planned=%s damage=%s transit=%s late_flag=%s event_id=%s",
        date_col, from_store_col, to_store_col, units_actual_col, units_planned_col, damage_col, transit_col, late_flag_col, event_id_col
    )

    # Basic validation: we need at least date + from/to to be meaningful
    if date_col is None:
        LOG.warning("Date column not detected. YearMonth will be NA.")
    if from_store_col is None or to_store_col is None:
        LOG.warning("From/To store columns not detected. Aggregation will be limited.")

    # Type conversion for metrics
    for c in [units_actual_col, units_planned_col, damage_col, transit_actual_col, transit_planned_col]:
        to_numeric(fle, c)

    # Late flag normalization
    if late_flag_col and late_flag_col in fle.columns:
        fle["Late_Flag_Bin"] = normalize_bool01(fle[late_flag_col])
    else:
        fle["Late_Flag_Bin"] = np.nan

    # Attach YearMonth
    if date_col and date_col in fle.columns:
        fle = fle.merge(dim_date.rename(columns={"Date_ID": date_col}), on=date_col, how="left")
    else:
        fle["YearMonth"] = pd.NA

    # Attach store attributes (From / To)
    if from_store_col and from_store_col in fle.columns:
        fle = fle.merge(
            dim_store.rename(columns={
                "Store_ID": from_store_col,
                "Store_Type": "From_Store_Type",
                "Country": "From_Country",
                "Region": "From_Region",
                "Is_Store": "From_Is_Store",
                "Is_DC": "From_Is_DC",
            }),
            on=from_store_col,
            how="left",
        )
    else:
        fle["From_Is_Store"] = np.nan
        fle["From_Is_DC"] = np.nan

    if to_store_col and to_store_col in fle.columns:
        fle = fle.merge(
            dim_store.rename(columns={
                "Store_ID": to_store_col,
                "Store_Type": "To_Store_Type",
                "Country": "To_Country",
                "Region": "To_Region",
                "Is_Store": "To_Is_Store",
                "Is_DC": "To_Is_DC",
            }),
            on=to_store_col,
            how="left",
        )
    else:
        fle["To_Is_Store"] = np.nan
        fle["To_Is_DC"] = np.nan

    # Normalize Is_Store / Is_DC to numeric 0/1 where possible
    for c in ["From_Is_Store", "From_Is_DC", "To_Is_Store", "To_Is_DC"]:
        if c in fle.columns:
            fle[c] = pd.to_numeric(fle[c], errors="coerce")

    # Flow type classification (vectorized; avoid row-wise apply)
    from_is_dc = fle.get("From_Is_DC")
    to_is_dc = fle.get("To_Is_DC")
    from_is_store = fle.get("From_Is_Store")
    to_is_store = fle.get("To_Is_Store")

    # Supplier -> DC heuristic: from_store_id is NA AND to_is_dc == 1
    if from_store_col and from_store_col in fle.columns:
        from_id_na = fle[from_store_col].isna()
    else:
        from_id_na = pd.Series(True, index=fle.index)

    if to_store_col and to_store_col in fle.columns:
        to_id_ok = fle[to_store_col].notna()
    else:
        to_id_ok = pd.Series(False, index=fle.index)

    cond_supplier_to_dc = from_id_na & to_id_ok & (to_is_dc == 1)
    cond_dc_to_store = (from_is_dc == 1) & (to_is_store == 1)
    cond_store_to_dc = (from_is_store == 1) & (to_is_dc == 1)
    cond_store_to_store = (from_is_store == 1) & (to_is_store == 1)

    flow_type = np.select(
        [cond_supplier_to_dc, cond_dc_to_store, cond_store_to_dc, cond_store_to_store],
        ["Supplier_to_DC", "DC_to_Store", "Store_to_DC", "Store_to_Store"],
        default="Other",
    )
    fle["Flow_Type"] = flow_type

    # Choose volume column
    volume_col = units_actual_col or units_planned_col
    if volume_col is None:
        volume_col = "Units_Fallback"
        fle[volume_col] = 1.0

    group_cols = ["YearMonth"]
    if from_store_col:
        group_cols.append(from_store_col)
    if to_store_col:
        group_cols.append(to_store_col)
    group_cols.append("Flow_Type")

    agg = {
        "Events_Count": (event_id_col, "count"),
        "Units_Moved_Total": (volume_col, "sum"),
        "Late_Share": ("Late_Flag_Bin", "mean"),
    }
    if units_planned_col:
        agg["Units_Planned_Total"] = (units_planned_col, "sum")
    if damage_col:
        agg["Damage_Loss_Total"] = (damage_col, "sum")
    if transit_col and transit_col in fle.columns:
        agg["Transit_Days_Avg"] = (transit_col, "mean")

    mart_dc = (
        fle.groupby(group_cols, dropna=False)
        .agg(**agg)
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_mart = output_dir / "mart_dc_flows.csv"
    mart_dc.to_csv(out_mart, index=False, encoding="utf-8-sig")
    LOG.info("Saved %-28s shape=%s", out_mart.name, mart_dc.shape)

    # Optional: small summary outputs (kept, but not essential for the dashboard)
    summary = {
        "Events": ("Events_Count", "sum"),
        "Units_Moved": ("Units_Moved_Total", "sum"),
        "Late_Share_Avg": ("Late_Share", "mean"),
    }
    if "Damage_Loss_Total" in mart_dc.columns:
        summary["Damage_Loss_Total"] = ("Damage_Loss_Total", "sum")
    if "Transit_Days_Avg" in mart_dc.columns:
        summary["Transit_Days_Avg"] = ("Transit_Days_Avg", "mean")

    summary_flow = mart_dc.groupby("Flow_Type").agg(**summary).reset_index()
    out_summary = output_dir / "dc_flows_summary.csv"
    summary_flow.to_csv(out_summary, index=False, encoding="utf-8-sig")
    LOG.info("Saved %-28s shape=%s", out_summary.name, summary_flow.shape)

    dc_to_store = mart_dc[mart_dc["Flow_Type"] == "DC_to_Store"].copy()
    out_subset = output_dir / "dc_flows_dc_to_store.csv"
    dc_to_store.to_csv(out_subset, index=False, encoding="utf-8-sig")
    LOG.info("Saved %-28s shape=%s", out_subset.name, dc_to_store.shape)

    return mart_dc


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mart_dc_flows.csv from fact_logistics_events + dims.")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Folder with raw CSV files.")
    p.add_argument("--output-dir", type=Path, default=Path("data/marts"), help="Folder to write output marts.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    build_dc_flows(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
