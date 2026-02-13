#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# parsing + utils (same logic as your 5-group scripts)
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

    # QK (model empty)
    if m == "":
        if p == "DFT_ND_RADIAL":
            return "QK_DFT"
        if p == "PCA_MATCHED":
            return "QK_PCA"
        if p == "RP_MATCHED":
            return "QK_RP"
        return None

    # classical SVM baselines: require preproc==classical_preproc (same policy as your other script)
    cp = str(classical_preproc).strip().upper()
    if m == "SVM_LINEAR" and p == cp:
        return "SVM_LINEAR"
    if m == "SVM_RBF" and p == cp:
        return "SVM_RBF"
    return None


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
# strict completeness
# -----------------------------
def strict_complete_curves(long: pd.DataFrame, sigmas_required: np.ndarray) -> pd.DataFrame:
    complete_keys = []
    for (ds, grp, cfg), sub in long.groupby(["dataset", "group", "config_id"]):
        svals = np.sort(sub["noise_sigma"].to_numpy(float))
        if all(np.any(np.isclose(svals, sr)) for sr in sigmas_required):
            complete_keys.append((ds, grp, cfg))
    if not complete_keys:
        raise SystemExit("No complete curves found under strict policy.")
    keep = set(complete_keys)
    return long[long.apply(lambda r: (r["dataset"], r["group"], r["config_id"]) in keep, axis=1)].copy()


# -----------------------------
# sigma0 tuning: choose best config per (dataset, group), then freeze
# -----------------------------
def sigma0_tune_and_freeze(long: pd.DataFrame, sigma0: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    s0 = long[np.isclose(long["noise_sigma"].to_numpy(float), float(sigma0))].copy()
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
    return long_best, idx_best


# -----------------------------
# curves: (dataset, group, sigma) -> mean acc (should be unique anyway)
# -----------------------------
def build_frozen_curves(long_best: pd.DataFrame) -> pd.DataFrame:
    curves = (
        long_best.groupby(["dataset", "group", "noise_sigma"], as_index=False)
                 .agg(acc=("acc", "mean"))
                 .sort_values(["dataset", "group", "noise_sigma"])
                 .reset_index(drop=True)
    )
    return curves


def plot_dataset_curve(curves: pd.DataFrame, dataset: str, outdir: Path, sigmas_order: np.ndarray, groups_order: list[str]):
    ds = norm_dataset_name(dataset)
    sub = curves[curves["dataset"] == ds].copy()
    if sub.empty:
        print(f"[WARN] dataset not found: {dataset} (norm={ds})")
        return

    plt.figure(figsize=(7.2, 4.4))
    for g in groups_order:
        sg = sub[sub["group"] == g].copy()
        if sg.empty:
            continue
        sg = sg.set_index("noise_sigma").reindex(sigmas_order).reset_index()
        plt.plot(sg["noise_sigma"].values, sg["acc"].values, marker="o", label=g)

    plt.xlabel("Noise level σ")
    plt.ylabel("Accuracy (sigma0-tuned & frozen config)")
    plt.title(f"{ds} — accuracy vs noise (frozen @ σ0)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    fn = outdir / f"curve_frozen_{ds.replace('/', '_')}.png"
    plt.savefig(fn, dpi=220)
    plt.close()


def plot_global_curve(curves: pd.DataFrame, outdir: Path, sigmas_order: np.ndarray, groups_order: list[str]):
    g = (
        curves.groupby(["group", "noise_sigma"], as_index=False)
              .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"), n_ds=("acc", "size"))
              .sort_values(["group", "noise_sigma"])
              .reset_index(drop=True)
    )

    plt.figure(figsize=(7.4, 4.6))
    for grp in groups_order:
        sg = g[g["group"] == grp].copy()
        if sg.empty:
            continue
        sg = sg.set_index("noise_sigma").reindex(sigmas_order).reset_index()
        plt.plot(sg["noise_sigma"].values, sg["acc_mean"].values, marker="o", label=grp)

    plt.xlabel("Noise level σ")
    plt.ylabel("Accuracy (avg over datasets; frozen @ σ0)")
    plt.title("Global degradation (sigma0-tuned & frozen; averaged over datasets)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    fn = outdir / "curve_frozen_global_by_group.png"
    plt.savefig(fn, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results_dft_vs_pca_vs_rp_real_2D")
    ap.add_argument("--pattern", type=str, default="rp_real_2d_color_*.csv")
    ap.add_argument("--acc_col", type=str, default="accuracy_mean")
    ap.add_argument("--sigmas", type=str, default="0.000,0.025,0.050,0.075,0.100,0.125,0.150,0.200")
    ap.add_argument("--sigma0", type=float, default=0.0, help="Noise used to tune (select best config).")
    ap.add_argument("--classical_preproc", type=str, default="DFT_ND_RADIAL")
    ap.add_argument("--outdir", type=str, default="curves_5groups_sigma0_tuned")
    ap.add_argument("--datasets", type=str, default="digits,cifar10,olivetti,pneumoniamnist",
                    help="Comma-separated list; if empty, take first 4 found.")
    ap.add_argument("--no_strict", action="store_true", help="Disable strict completeness filtering.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sigmas_required = np.sort(np.array([float(x) for x in args.sigmas.split(",")], float))
    groups_order = ["QK_DFT", "QK_PCA", "QK_RP", "SVM_LINEAR", "SVM_RBF"]

    long, diag = build_long(Path(args.results_dir), args.pattern, args.acc_col, args.classical_preproc)
    print("[DIAG_RAW]", diag)
    if long.empty:
        raise SystemExit("No data after grouping.")

    # keep sigma grid
    long = long[long["noise_sigma"].apply(lambda s: np.any(np.isclose(float(s), sigmas_required)))].copy()
    if long["noise_sigma"].nunique() < 2:
        raise SystemExit("Not enough sigma points after filtering.")

    if not args.no_strict:
        long = strict_complete_curves(long, sigmas_required)

    # sigma0 tune + freeze
    long_best, idx_best = sigma0_tune_and_freeze(long, float(args.sigma0))

    # curves
    curves = build_frozen_curves(long_best)
    curves.to_csv(outdir / "curves_frozen_by_dataset.csv", index=False)
    long_best.to_csv(outdir / "long_best_sigma0_5groups.csv", index=False)
    idx_best.to_csv(outdir / "best_config_per_dataset_group_sigma0.csv", index=False)

    # datasets to plot
    if args.datasets.strip():
        ds_list = [norm_dataset_name(x.strip()) for x in args.datasets.split(",") if x.strip()]
    else:
        ds_list = sorted(curves["dataset"].unique().tolist())[:4]

    for ds in ds_list:
        plot_dataset_curve(curves, ds, outdir, sigmas_required, groups_order)

    plot_global_curve(curves, outdir, sigmas_required, groups_order)

    print("[OK] Saved:")
    print(f"  {(outdir / 'curves_frozen_by_dataset.csv').resolve()}")
    print(f"  {(outdir / 'curve_frozen_global_by_group.png').resolve()}")
    print(f"  (plus per-dataset curve_frozen_*.png in {outdir.resolve()})")


if __name__ == "__main__":
    main()
