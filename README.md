# Power BI: Inventory & Procurement Control Tower (2016–2026)

---

## Overview

![Dashboard Preview]([Resources/P3.gif](https://github.com/Q-Lesh/PowerBi_Project_3/blob/main/Resourses/P3.gif?raw=true))

This repository contains a five-page executive-level Power BI control tower built on a structured retail dataset (2016–2026).

The project covers an enterprise inventory and procurement governance environment focused on:

- Capital efficiency  
- Supplier reliability  
- Distribution network stability  
- Structural risk exposure  

The solution is designed for board-level decision support rather than operational reporting.  
It prioritizes risk concentration, imbalance detection, and intervention signals over transactional detail.

**The objective is to surface structural weaknesses early and support capital allocation decisions at management level.**

---

## Business Context

Retail inventory sits at the intersection of:

- Working capital allocation  
- Service level stability  
- Supplier execution quality  
- Distribution network performance  

Structural risks addressed:

- **Overstock (>180d coverage)** → trapped capital  
- **Zero-stock exposure** → service degradation  
- **Supplier underfill & delays** → replenishment instability  
- **DC bottlenecks** → network inefficiency  
- **Execution imbalance** → **profitability pressure and inefficient capital rotation**

The objective is intervention prioritization — not reporting.

---

## Dataset

**Time range:** January 2016 – January 2026  

### Data Characteristics

- Stable multi-year evergreen SKUs  
- Anonymized product & supplier names  
- Intentionally distorted pricing  
- Controlled naming noise for realism  
- **Executive KPIs intentionally anchored to 2025 to simulate a fixed annual governance snapshot**
- **Date slicers are present for structural consistency, but executive KPIs are deliberately hard-anchored by design**

---

## Download Dashboard & Dataset

Full project package available under Releases:

https://github.com/Q-Lesh/PowerBI_Project_3/releases/tag/v1.0

Includes:

- `Procurement_Control_Tower_v1.0.pbix`
- `data/` (analytical marts)
- `data/raw/` (source files)
- `Python build & QA scripts` (some examples)

After extraction, update file paths in Power BI Desktop before refreshing.

---

# Dashboard Pages

Five structured pages forming a decision making control tool.

---

### 1. Inventory Health

![Inventory Health](Resources/P1.gif)

**Purpose:**  
Evaluate structural inventory balance.

**Focus:**
- EOM inventory trend  
- Zero-stock exposure  
- Low coverage (<30d)  
- Overstock share (>180d)  
- **Share of active SKU–store positions used as denominator for risk KPIs**

**Decision Enabled:**
- Capital reallocation  
- Risk-heavy category intervention  
- Availability stabilization  

---

### 2. Replenishment & PO Performance

![Procurement Performance](Resources/P2.gif)

**Purpose:**  
Assess procurement execution reliability.

**Focus:**
- PO Fill Rate (FY2025)  
- Underfill exposure  
- Underfill by subcategory  
- Supplier concentration of risk  

**Decision Enabled:**
- Escalation of chronic underperformance  
- Supplier allocation adjustment  
- Contract renegotiation basis  

---

### 3. Supplier Risk Scorecard

![Supplier Risk](Resources/P3.gif)

**Purpose:**  
Classify structural supplier risk.

**Focus:**
- Composite Risk Score  
- Reliability vs Delay matrix  
- Fill Rate & Late % distribution  
- High-risk supplier ranking  

**Decision Enabled:**
- Diversification strategy  
- Dependency risk reduction  
- Sourcing portfolio balancing  

---

### 4. Distribution Network Risk & Performance

![DC Risk](Resources/P4.gif)

**Purpose:**  
Evaluate logistics network stability.

**Focus:**
- Average transit time  
- SLA breach share  
- Late trend  
- Destination exposure  

**Decision Enabled:**
- Bottleneck identification  
- SLA review  
- Routing optimization  

---

### 5. Executive Overview — Inventory & Supply Risk

![Executive Overview](Resources/P5.gif)

**Purpose:**  
Board-level structural risk snapshot.

**Core KPI Cards:**
- Capital trapped (>180d)
- Service at risk (Zero stock)
- Supplier execution (Fill Rate)
- Network instability (DC SLA breach)

Designed for a <20-second executive scan.

**KPIs intentionally fixed to FY2025 to simulate an annual executive risk review window rather than rolling operational monitoring.**

Unstable rolling coverage logic intentionally excluded.

---

# Data Architecture

### Analytical Marts

- `mart_inventory_monthly`
- `mart_bridge_sales_inventory_po`
- `mart_supplier_scorecard`
- `mart_dc_flows`

### Dimensions

- `dim_date`
- `dim_product`
- `dim_store`
- `dim_supplier`
- `dim_channel`
- `dim_currency`

### Modeling Rules

- Single-direction relationships  
- No ambiguous joins  
- No calculated columns for core KPIs  
- DAX-only metric logic  
- Grain consistency enforced  
- **All risk measures calculated at explicit grain (Product × Store × Month or Supplier × Year)**

---

# Data Engineering & QA Layer (Python)

All analytical marts are constructed and validated using a structured Python pipeline.

The objective of the Python layer is not visualization, but:

- Controlled aggregation  
- Deterministic reproducibility  
- Grain enforcement  
- Structural validation before semantic modeling  

---

## Build Layer

Build scripts construct analytical marts from raw fact tables.

Key principles implemented:

- **Explicit aggregation logic (no implicit groupings)**  
- **Strict grain control (Product × Store × YearMonth enforced for bridge mart)**  
- **Vectorized transformations (no row-wise apply logic)**  
- **Deterministic transformations — no uncontrolled randomness**  
- **Supplier scorecard calculated via yearly aggregation of PO + Invoice raw facts**  
- **DC flows classified using vectorized flow-type logic (Supplier → DC, DC → Store, etc.)**

The build stage ensures Power BI receives structurally clean, governance-ready marts rather than operational raw data.

---

## QA Layer

`qa_integrity_checks.py` performs structured validation before dashboard consumption.

Validation stages:

1. **PK/FK integrity checks across all fact-dimension relationships**
2. **Orphan key detection**
3. **Duplicate grain detection in marts**
4. **Row-count reconciliation (raw vs mart)**
5. **Metric reconciliation (raw aggregates vs mart values)**
6. **Inventory balance equation validation**
7. **Tolerance-based mismatch detection (with fail-on-warning mode)**

The QA layer ensures:

- No silent duplication  
- No broken joins  
- No aggregation drift  
- No grain distortion  

---

## Diagnostics Layer

Diagnostic scripts generate structural summaries to validate behavioral realism.

Examples:

- Supplier fill-rate dispersion  
- Late percentage distribution  
- Reliability score banding  
- Coverage band concentration  

These diagnostics are not part of the dashboard itself but serve as structural verification tools to validate data behavior before modeling.

---

# Technical Stack

- Power BI  
- DAX  
- Python (data engineering & validation)  
- Structured CSV marts  

Pipeline:

Raw files → Python build layer → QA validation → Analytical marts → Power BI semantic model

---

# Real-World Extension

In production, this architecture could evolve into:

- Rolling 12M governance logic  
- ERP-integrated refresh  
- Automated SLA breach alerts  
- Contract penalty modeling  
- Cash-flow simulation  
- Forecast integration layer  

---

# Conclusion

This project demonstrates how disciplined modeling, controlled aggregation, and structured risk layering can transform retail operational data into a solution designed to support management decision-making.

It emphasizes:

- Reproducibility  
- Structural discipline  
- Explicit grain control  
- Executive clarity  
- Intervention prioritization  
