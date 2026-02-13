#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wild cluster bootstrap (by dataset) with *sigma0 hyperparameter selection* for:
  QK_DFT vs QK_PCA vs QK_RP vs SVM_LINEAR vs SVM_RBF

Protocol (paper-defensible, pro-QK without cheating):
  1) Build long with groups
  2) Strict completeness over required sigmas for each (dataset, group, config_id)
  3) For each (dataset, group), pick the BEST config at sigma0 (highest mean acc at sigma0)
  4) Freeze those configs, and run the SAME regression+wild cluster bootstrap as before

Model (dataset FE + slope interactions), base group = QK_DFT:
  acc = dataset_FE + beta_sigma*sigma + Σ_{g≠base} beta_g * sigma * I(group=g) + eps
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# parsing + utils
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
    for c in cols:
        if str(c).strip().lower() in ["", "unnamed: 0", "unnamed: 1", "unnamed: 2"]:
            return c
    return None


def group_from_row(preproc: str, model_norm: str, classical_preproc: str) -> Optional[str]:
    p = str(preproc).strip().upper()
    m = str(model_norm).strip().upper()

    if m == "":
        if p == "DFT_ND_RADIAL":
            return "QK_DFT"
        if p == "PCA_MATCHED":
            return "QK_PCA"
        if p == "RP_MATCHED":
            return "QK_RP"
        return None

    cp = str(classical_preproc).strip().upper()
    if m == "SVM_LINEAR" and p == cp:
        return "SVM_LINEAR"
    if m == "SVM_RBF" and p == cp:
        return "SVM_RBF"
    return None


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
# OLS
# -----------------------------
def ols_fit(X: np.ndarray, y: np.ndarray):
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XtX_inv


# -----------------------------
# Design matrix (dataset FE), base group = QK_DFT
# -----------------------------
def build_global_design(long: pd.DataFrame):
    df = long.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["group"] = df["group"].astype(str)
    df["sigma"] = df["noise_sigma"].astype(float)
    df["acc"] = df["acc"].astype(float)

    datasets = sorted(df["dataset"].unique().tolist())
    base_ds = datasets[0]
    for ds in datasets[1:]:
        df[f"FE_{ds}"] = (df["dataset"] == ds).astype(float)

    base = "QK_DFT"
    groups = ["QK_DFT", "QK_PCA", "QK_RP", "SVM_LINEAR", "SVM_RBF"]
    for g in groups:
        df[f"I_{g}"] = (df["group"] == g).astype(float)

    df["const"] = 1.0

    inter_cols = {}
    for g in groups:
        if g == base:
            continue
        col = f"sigma_x_{g}"
        df[col] = df["sigma"] * df[f"I_{g}"]
        inter_cols[g] = col

    X_cols = ["const"] + [f"FE_{ds}" for ds in datasets[1:]] + ["sigma"] + list(inter_cols.values())
    X = df[X_cols].to_numpy(float)
    y = df["acc"].to_numpy(float)
    clusters = df["dataset"].to_numpy()

    idx_sigma = X_cols.index("sigma")
    idx_inter = {g: X_cols.index(col) for g, col in inter_cols.items()}

    meta = {
        "X_cols": X_cols,
        "datasets": datasets,
        "base_ds_dropped": base_ds,
        "groups": groups,
        "base_group": base,
        "idx_sigma": idx_sigma,
        "idx_inter": idx_inter,
    }
    return df, X, y, clusters, meta


def betas_to_slopes(beta: np.ndarray, meta: Dict) -> Dict[str, float]:
    base = meta["base_group"]
    groups = meta["groups"]
    i0 = meta["idx_sigma"]
    idx_inter = meta["idx_inter"]

    out = {}
    out[f"slope_{base}"] = float(beta[i0])
    for g in groups:
        if g == base:
            continue
        out[f"slope_{g}"] = float(beta[i0] + beta[idx_inter[g]])
        out[f"delta_{g}_minus_{base}"] = float(beta[idx_inter[g]])
    return out


# -----------------------------
# Curves: balanced mean over dataset×config curves
# -----------------------------
def mean_curve_over_curves(df: pd.DataFrame, sigmas: np.ndarray, group: str, acc_col: str) -> np.ndarray:
    req = np.sort(np.asarray(sigmas, float))
    d = df[df["group"] == group].copy()
    if d.empty:
        return np.full(req.size, np.nan, float)

    curves = []
    for (ds, cfg), g in d.groupby(["dataset", "config_id"]):
        s = g["noise_sigma"].to_numpy(float)
        a = g[acc_col].to_numpy(float)

        a_on = np.full(req.size, np.nan, float)
        for i, sr in enumerate(req):
            j = np.where(np.isclose(s, sr))[0]
            a_on[i] = float(a[j[0]]) if j.size else np.nan

        if np.all(np.isfinite(a_on)):
            curves.append(a_on)

    if not curves:
        return np.full(req.size, np.nan, float)

    return np.vstack(curves).mean(axis=0)


# -----------------------------
# Wild cluster bootstrap
# -----------------------------
def wild_bootstrap_betas_and_curves(df_design, X, y, clusters, meta, sigmas, B, seed):
    rng = np.random.default_rng(seed)

    beta_hat, resid_hat, _ = ols_fit(X, y)
    yhat = X @ beta_hat

    uniq = pd.unique(clusters)
    idx_map = {g: np.where(clusters == g)[0] for g in uniq}

    slopes_hat = betas_to_slopes(beta_hat, meta)
    boot_slopes = {k: np.empty(B, float) for k in slopes_hat.keys()}

    groups = meta["groups"]
    boot_curves = {g: np.empty((B, len(sigmas)), float) for g in groups}

    df0 = df_design.copy().reset_index(drop=True)

    for b in range(B):
        w = {g: (1.0 if rng.integers(0, 2) == 1 else -1.0) for g in uniq}
        y_star = yhat.copy()
        for g in uniq:
            idx = idx_map[g]
            y_star[idx] = yhat[idx] + w[g] * resid_hat[idx]

        beta_star, _, _ = ols_fit(X, y_star)
        slopes_star = betas_to_slopes(beta_star, meta)
        for k in boot_slopes:
            boot_slopes[k][b] = float(slopes_star[k])

        df_star = df0.copy()
        df_star["acc_star"] = y_star
        for g in groups:
            boot_curves[g][b, :] = mean_curve_over_curves(df_star, sigmas, g, "acc_star")

    return slopes_hat, boot_slopes, boot_curves


# -----------------------------
# Plots
# -----------------------------
def plot_betas_bar(boot_slopes, slopes_hat, meta, outpath: Path):
    groups = meta["groups"]
    keys = [f"slope_{g}" for g in groups]

    est = np.array([slopes_hat[k] for k in keys], float)
    lo = np.array([np.quantile(boot_slopes[k], 0.025) for k in keys], float)
    hi = np.array([np.quantile(boot_slopes[k], 0.975) for k in keys], float)
    yerr = np.vstack([est - lo, hi - est])

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(groups))
    ax.bar(x, est)
    ax.errorbar(x, est, yerr=yerr, fmt="none", capsize=6, linewidth=1.5)
    ax.axhline(0.0, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Slope β (accuracy change per +1.0 σ)")
    ax.set_title("Noise sensitivity (after sigma0 tuning, 95% CI)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_curves(boot_curves, sigmas, meta, outpath: Path):
    groups = meta["groups"]
    req = np.sort(np.asarray(sigmas, float))

    fig = plt.figure(figsize=(10.8, 5.5))
    ax = fig.add_subplot(111)

    for g in groups:
        C = boot_curves[g]
        med = np.nanmedian(C, axis=0)
        lo = np.nanquantile(C, 0.025, axis=0)
        hi = np.nanquantile(C, 0.975, axis=0)
        ax.plot(req, med, label=g)
        ax.fill_between(req, lo, hi, alpha=0.2)

    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs noise (after sigma0 tuning): median ± bootstrap 95% band")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results_dft_vs_pca_vs_rp_real_2D")
    ap.add_argument("--pattern", type=str, default="rp_real_2d_color_*.csv")
    ap.add_argument("--acc_col", type=str, default="accuracy_mean")
    ap.add_argument("--sigmas", type=str, default="0.000,0.025,0.050,0.075,0.100,0.125,0.150,0.200")
    ap.add_argument("--sigma0", type=float, default=0.0, help="Noise level used for config selection (tuning).")
    ap.add_argument("--classical_preproc", type=str, default="DFT_ND_RADIAL")
    ap.add_argument("--outdir", type=str, default="robustness_all_groups_best_sigma0")
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sigmas_required = np.sort(np.array([float(x) for x in args.sigmas.split(",")], float))

    long, diag = build_long(Path(args.results_dir), args.pattern, args.acc_col, args.classical_preproc)
    print("[DIAG_RAW]", diag)
    if long.empty:
        raise SystemExit("No data after grouping.")

    # keep required sigmas
    long = long[long["noise_sigma"].apply(lambda s: np.any(np.isclose(float(s), sigmas_required)))].copy()

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

    # ---------
    # sigma0 tuning: best config per (dataset, group)
    # ---------
    sigma0 = float(args.sigma0)
    s0 = long[np.isclose(long["noise_sigma"].to_numpy(float), sigma0)].copy()
    if s0.empty:
        raise SystemExit(f"No rows at sigma0={sigma0:.3f} after filtering.")

    # mean acc at sigma0 (in case you have duplicates / multiple seeds)
    s0_mean = (
        s0.groupby(["dataset", "group", "config_id"], as_index=False)["acc"]
          .mean()
          .rename(columns={"acc": "acc_sigma0"})
    )

    # pick best config_id per (dataset, group)
    idx_best = s0_mean.sort_values(["dataset", "group", "acc_sigma0"], ascending=[True, True, False]) \
                     .groupby(["dataset", "group"], as_index=False) \
                     .head(1)

    best_map = {(r["dataset"], r["group"]): r["config_id"] for _, r in idx_best.iterrows()}

    long_best = long[long.apply(lambda r: r["config_id"] == best_map.get((r["dataset"], r["group"])), axis=1)].copy()

    diag_best = {
        "rows_best": int(len(long_best)),
        "datasets": int(long_best["dataset"].nunique()),
        "groups": {g: int((long_best["group"] == g).sum() / len(sigmas_required)) for g in sorted(long_best["group"].unique())},
        "configs_per_group": long_best.groupby("group")["config_id"].nunique().to_dict(),
        "sigma0": sigma0,
    }
    print("[DIAG_BEST]", diag_best)

    long_best.to_csv(outdir / "long_best_sigma0_all_groups.csv", index=False)
    idx_best.to_csv(outdir / "best_config_per_dataset_group_sigma0.csv", index=False)

    # regression + bootstrap
    df_design, X, y, clusters, meta = build_global_design(long_best)
    slopes_hat, boot_slopes, boot_curves = wild_bootstrap_betas_and_curves(
        df_design, X, y, clusters, meta, sigmas_required, int(args.B), int(args.seed)
    )

    print("\n[HAT] slope β estimates (accuracy change per +1.0 σ):")
    for g in meta["groups"]:
        print(f"  β_{g:10s} = {slopes_hat[f'slope_{g}']:+.6f}")
    base = meta["base_group"]
    for g in meta["groups"]:
        if g == base:
            continue
        print(f"  Δ({g}−{base}) = {slopes_hat[f'delta_{g}_minus_{base}']:+.6f}")

    pd.DataFrame(boot_slopes).to_csv(outdir / "bootstrap_slopes_draws_best_sigma0.csv", index=False)

    # summary
    rows = []
    for k, hat in slopes_hat.items():
        draws = boot_slopes[k]
        lo, hi = np.quantile(draws, [0.025, 0.975])
        rows.append({"param": k, "hat": float(hat), "ci95_lo": float(lo), "ci95_hi": float(hi), "Pr_lt_0": float(np.mean(draws < 0.0))})
    pd.DataFrame(rows).to_csv(outdir / "robustness_summary_best_sigma0.csv", index=False)

    # plots
    plot_betas_bar(boot_slopes, slopes_hat, meta, outdir / "F_betas_bar_best_sigma0.png")
    plot_curves(boot_curves, sigmas_required, meta, outdir / "F_curves_best_sigma0.png")

    print("\n[ok] Saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
