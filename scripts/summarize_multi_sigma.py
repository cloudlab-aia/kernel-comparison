#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Winner counts vs noise sigma for 5 groups:
  QK_DFT vs QK_PCA vs QK_RP vs SVM_LINEAR vs SVM_RBF

Protocol (paper-defensible, consistent with sigma0 tuning script):
  1) Build long with groups (same grouping logic as sigma0 bootstrap script)
  2) Strict completeness over required sigmas for each (dataset, group, config_id)
  3) For each (dataset, group), pick the BEST config at sigma0
  4) Freeze those configs and compute winner per (dataset, sigma)
  5) Plot winner counts vs sigma (datasets are the unit)

Notes:
- "winner" is computed among the 5 groups with a tolerance; ties -> DRAW.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# parsing + utils (same as your sigma0 bootstrap script)
# -----------------------------
def norm_dataset_name(s: str) -> str:
    s = str(s).strip().lower()
    s = (
        s.replace("emnist_byclass", "emnist/byclass")
         .replace("emnist-byclass", "emnist/byclass")
         .replace("emnistbyclass", "emnist/byclass")
    )
    for suf in ["_real", "_color", "_grey", "_gray"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def parse_dataset_sigma_from_filename(fp: Path) -> Tuple[Optional[str], Optional[float]]:
    stem = fp.stem
    prefix = "rp_real_2d_color_"
    if not stem.startswith(prefix):
        return None, None
    core = stem[len(prefix):]
    if core.endswith("_v5"):
        core = core[:-3]
    parts = core.split("_")
    if len(parts) < 2:
        return None, None
    sigma_str = parts[-1]
    dataset = "_".join(parts[:-1])
    try:
        sigma = float(sigma_str)
    except ValueError:
        return None, None
    return norm_dataset_name(dataset), float(sigma)


def cfg_id(row) -> str:
    return f"nq={int(row['n_qubits'])}|B={int(row['budget'])}|k={int(row['k'])}|m={int(row['m_used'])}"


def normalize_model(model_val) -> str:
    if model_val is None:
        return ""
    if isinstance(model_val, float) and np.isnan(model_val):
        return ""
    s = str(model_val).strip().upper()
    if s in ["", "NONE", "NAN", "NULL"]:
        return ""
    if "SVM" in s and "LINEAR" in s:
        return "SVM_LINEAR"
    if "SVM" in s and "RBF" in s:
        return "SVM_RBF"
    return s


def find_model_column(cols) -> Optional[str]:
    cols = list(cols)
    if "model" in cols:
        return "model"
    # fallback: some runs put the model name in an unnamed col
    for c in cols:
        if str(c).strip().lower() in ["", "unnamed: 0", "unnamed: 1", "unnamed: 2"]:
            return c
    return None


def group_from_row(preproc: str, model_norm: str, classical_preproc: str) -> Optional[str]:
    p = str(preproc).strip().upper()
    m = str(model_norm).strip().upper()

    # QK rows typically have empty/None model
    if m == "":
        if p == "DFT_ND_RADIAL":
            return "QK_DFT"
        if p == "PCA_MATCHED":
            return "QK_PCA"
        if p == "RP_MATCHED":
            return "QK_RP"
        return None

    # classical SVM baselines: preproc must match classical_preproc (same policy as your script)
    cp = str(classical_preproc).strip().upper()
    if m == "SVM_LINEAR" and p == cp:
        return "SVM_LINEAR"
    if m == "SVM_RBF" and p == cp:
        return "SVM_RBF"
    return None


def compute_winner_5way(row: pd.Series, groups, tol: float) -> str:
    vals = {}
    for g in groups:
        v = row.get(g, np.nan)
        if pd.notna(v) and np.isfinite(v):
            vals[g] = float(v)
    if not vals:
        return "NA"
    max_val = max(vals.values())
    winners = [k for k, v in vals.items() if abs(v - max_val) <= tol]
    return winners[0] if len(winners) == 1 else "DRAW"


# -----------------------------
# build long with groups
# -----------------------------
def build_long(results_dir: Path, pattern: str, acc_col: str, classical_preproc: str):
    files = sorted(results_dir.glob(pattern))
    rows = []
    bad_parse = bad_cols = empty_clean = 0
    dropped_not_in_group = 0

    for fp in files:
        ds_file, sigma = parse_dataset_sigma_from_filename(fp)
        if ds_file is None:
            bad_parse += 1
            continue

        df = pd.read_csv(fp)

        req = {"preproc", "m_used", acc_col, "n_qubits", "budget", "k"}
        if not req.issubset(df.columns):
            bad_cols += 1
            continue

        df = df.copy()

        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].map(norm_dataset_name)
            dataset = df["dataset"].iloc[0] if df["dataset"].nunique() == 1 else ds_file
        else:
            dataset = ds_file

        for c in ["m_used", "n_qubits", "budget", "k"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")

        mcol = find_model_column(df.columns)
        if mcol is None:
            df["model_norm"] = ""
        else:
            df["model_norm"] = df[mcol].apply(normalize_model)

        df["group"] = df.apply(lambda r: group_from_row(r["preproc"], r["model_norm"], classical_preproc), axis=1)

        before = len(df)
        df = df.dropna(subset=[acc_col, "n_qubits", "budget", "k", "m_used", "preproc"])
        df = df[df["group"].notna()]
        dropped_not_in_group += (before - len(df))

        if len(df) == 0:
            empty_clean += 1
            continue

        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "noise_sigma": float(sigma),
                    "group": str(r["group"]),
                    "config_id": cfg_id(r),
                    "acc": float(r[acc_col]),
                }
            )

    long = pd.DataFrame(rows)
    diag = {
        "files_matched": len(files),
        "bad_parse": bad_parse,
        "bad_cols": bad_cols,
        "empty_after_clean": empty_clean,
        "rows": len(long),
        "datasets": int(long["dataset"].nunique()) if not long.empty else 0,
        "sigmas_found": sorted(long["noise_sigma"].unique().tolist()) if not long.empty else [],
        "groups_found": sorted(long["group"].unique().tolist()) if not long.empty else [],
        "dropped_not_in_group_rows": int(dropped_not_in_group),
        "classical_preproc": str(classical_preproc),
    }
    return long, diag


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results_dft_vs_pca_vs_rp_real_2D")
    ap.add_argument("--pattern", type=str, default="rp_real_2d_color_*.csv")
    ap.add_argument("--acc_col", type=str, default="accuracy_mean")
    ap.add_argument("--sigmas", type=str, default="0.000,0.025,0.050,0.075,0.100,0.125,0.150,0.200")
    ap.add_argument("--sigma0", type=float, default=0.0, help="Noise used to tune (select best config).")
    ap.add_argument("--classical_preproc", type=str, default="DFT_ND_RADIAL")
    ap.add_argument("--outdir", type=str, default="summary_winners_5groups_best_sigma0")
    ap.add_argument("--winner_tol", type=float, default=1e-12, help="Tolerance to declare DRAW")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sigmas_required = np.sort(np.array([float(x) for x in args.sigmas.split(",")], float))
    groups = ["QK_DFT", "QK_PCA", "QK_RP", "SVM_LINEAR", "SVM_RBF"]

    long, diag = build_long(Path(args.results_dir), args.pattern, args.acc_col, args.classical_preproc)
    print("[DIAG_RAW]", diag)
    if long.empty:
        raise SystemExit("No data after grouping.")

    # keep only required sigmas
    long = long[long["noise_sigma"].apply(lambda s: np.any(np.isclose(float(s), sigmas_required)))].copy()
    if long["noise_sigma"].nunique() < 2:
        raise SystemExit("Not enough sigma points after filtering.")

    # strict completeness per (dataset, group, config_id)
    complete_keys = []
    for (ds, grp, cfg), sub in long.groupby(["dataset", "group", "config_id"]):
        svals = np.sort(sub["noise_sigma"].to_numpy(float))
        if all(np.any(np.isclose(svals, sr)) for sr in sigmas_required):
            complete_keys.append((ds, grp, cfg))
    if not complete_keys:
        raise SystemExit("No complete curves under strict policy.")
    keep = set(complete_keys)
    long = long[long.apply(lambda r: (r["dataset"], r["group"], r["config_id"]) in keep, axis=1)].copy()

    # sigma0 tuning: best config per (dataset, group)
    sigma0 = float(args.sigma0)
    s0 = long[np.isclose(long["noise_sigma"].to_numpy(float), sigma0)].copy()
    if s0.empty:
        raise SystemExit(f"No rows at sigma0={sigma0:.3f} after filtering.")

    s0_mean = (
        s0.groupby(["dataset", "group", "config_id"], as_index=False)["acc"]
          .mean()
          .rename(columns={"acc": "acc_sigma0"})
    )

    idx_best = (
        s0_mean.sort_values(["dataset", "group", "acc_sigma0"], ascending=[True, True, False])
              .groupby(["dataset", "group"], as_index=False)
              .head(1)
    )
    best_map = {(r["dataset"], r["group"]): r["config_id"] for _, r in idx_best.iterrows()}

    long_best = long[long.apply(lambda r: r["config_id"] == best_map.get((r["dataset"], r["group"])), axis=1)].copy()

    diag_best = {
        "rows_best": int(len(long_best)),
        "datasets": int(long_best["dataset"].nunique()),
        "sigma0": sigma0,
        "groups_found": sorted(long_best["group"].unique().tolist()),
        "configs_per_group": long_best.groupby("group")["config_id"].nunique().to_dict(),
    }
    print("[DIAG_BEST]", diag_best)

    # save chosen configs + long_best
    long_best.to_csv(outdir / "long_best_sigma0_5groups.csv", index=False)
    idx_best.to_csv(outdir / "best_config_per_dataset_group_sigma0.csv", index=False)

    # Build wide table per (dataset, sigma): one value per group (mean if duplicates)
    rows = []
    for (ds, sigma), sub in long_best.groupby(["dataset", "noise_sigma"]):
        vals = {}
        for g, subg in sub.groupby("group"):
            vals[g] = float(np.nanmean(subg["acc"].to_numpy(float)))
        row = {"dataset": ds, "noise_sigma": float(sigma)}
        for g in groups:
            row[g] = vals.get(g, np.nan)
        rows.append(row)

    wide = pd.DataFrame(rows)
    wide["winner"] = wide.apply(lambda r: compute_winner_5way(r, groups=groups, tol=float(args.winner_tol)), axis=1)

    # counts by sigma (datasets are the unit)
    win_counts = (
        wide.groupby(["noise_sigma", "winner"]).size()
            .unstack(fill_value=0)
            .reindex(index=np.sort(wide["noise_sigma"].unique()))
    )

    wide.to_csv(outdir / "winners_long_5groups_best_sigma0.csv", index=False)
    win_counts.to_csv(outdir / "win_counts_by_sigma_5groups_best_sigma0.csv")

    # Plot
    plt.figure(figsize=(8.2, 4.9))
    order_cols = [c for c in groups + ["DRAW", "NA"] if c in win_counts.columns]

    bottom = np.zeros(len(win_counts), dtype=float)
    x = np.arange(len(win_counts.index))

    for c in order_cols:
        vals = win_counts[c].to_numpy()
        plt.bar(x, vals, bottom=bottom, label=c)
        bottom += vals

    plt.xticks(x, [f"{v:.3f}".rstrip("0").rstrip(".") for v in win_counts.index], rotation=0)
    plt.xlabel("Noise σ")
    plt.ylabel("#datasets (winner counts)")
    plt.title("Winner counts vs noise (5 groups; sigma0-tuned; strict complete curves)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()

    outA = outdir / "F_winners_vs_sigma_5groups_best_sigma0.png"
    plt.savefig(outA, dpi=220)
    plt.close()

    print("\n[ok] Saved:")
    print(f"  {outA.resolve()}")
    print(f"  {(outdir / 'winners_long_5groups_best_sigma0.csv').resolve()}")
    print(f"  {(outdir / 'win_counts_by_sigma_5groups_best_sigma0.csv').resolve()}")
    print(f"  {(outdir / 'best_config_per_dataset_group_sigma0.csv').resolve()}")
    print(f"  {(outdir / 'long_best_sigma0_5groups.csv').resolve()}")


if __name__ == "__main__":
    main()
