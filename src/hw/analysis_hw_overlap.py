#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis of HW overlap benchmark:
  Methods: PES_DFT, ANGLE_raw, AE
  Metric: |Δp0| = |p0_hw - p0_expected|

Outputs:
  - Console summary
  - CSV tables in results/analysis_hw/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# Config
# =====================================================
csvs = ["results/hw_overlap_all_n2_fez.csv", "results/hw_overlap_all_n3_fez.csv", "results/hw_overlap_all_n4_fez.csv", 
        "results/hw_overlap_all_n2_marrakesh.csv", "results/hw_overlap_all_n3_marrakesh.csv", "results/hw_overlap_all_n4_marrakesh.csv"]
for  CSV_IN in csvs:
    #CSV_IN = "results/hw_overlap_all_n3_marrakesh.csv"
    OUTDIR = Path("results/analysis_hw")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Load data
    # =====================================================
    df = pd.read_csv(CSV_IN)

    """ print("\n=== Loaded data ===")
    print(df.head())
    print(f"\nTotal rows: {len(df)}") """
    print (CSV_IN)
    methods = sorted(df["method"].unique())
    rhos = sorted(df["rho"].unique())

    # =====================================================
    # Aggregate statistics per method
    # =====================================================
    agg = (
        df.groupby("method")
        .agg(
            mae_p0=("abs_err_p0", "mean"),
            std_p0=("abs_err_p0", "std"),
            max_err=("abs_err_p0", "max"),
            mean_depth=("depth", "mean"),
            mean_twoq=("twoq", "mean"),
            n_runs=("abs_err_p0", "count"),
        )
        .reset_index()
        .sort_values("mae_p0")
    )

    print("\n=== Aggregate error statistics (HW vs expected p0) ===")
    print(agg.to_string(index=False))

    agg.to_csv(OUTDIR / "table_aggregate_by_method.csv", index=False)
