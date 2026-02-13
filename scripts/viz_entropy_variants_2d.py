#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entropy visualization adapted to your *current* setup:
- variants are GAUSS with noise_sigma in {0.00, 0.10, 0.20} (not raw/blur/noise)
- everything is resized to out_hw=(32,32) (your current standard)
- uses load_real_2d_dataset() exactly like your pipeline
- produces a 3x3 grid: rows = 3 example images, cols = sigma levels
- prints H(sigma) per dataset

Save as: scripts/entropy_gauss_grid_viz.py
Run:
  python scripts/entropy_gauss_grid_viz.py
or:
  python scripts/entropy_gauss_grid_viz.py --datasets usps,optdigits,letter,vehicle,satimage --sigmas 0.00,0.10,0.20
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.classical.dataset_factory import load_dataset
from src.classical.real_2d_datasets import load_real_2d_dataset

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ============================================================
# Spectral entropy (same definition as your scan script)
# ============================================================
def power_spectrum_mean(X: np.ndarray) -> np.ndarray:
    """
    X: (N,H,W) float
    returns mean power spectrum (H,W)
    """
    X = X.astype(np.float64, copy=False)
    X = X - X.mean(axis=(1, 2), keepdims=True)
    F = np.fft.fftshift(np.fft.fft2(X, axes=(1, 2)), axes=(1, 2))
    P = (F.real ** 2 + F.imag ** 2)
    return P.mean(axis=0)


def spectral_entropy_norm(P: np.ndarray, eps: float = 1e-12) -> float:
    """
    Normalized Shannon entropy of normalized power spectrum, in [0,1]
    """
    v = np.asarray(P, dtype=np.float64).ravel()
    tot = float(v.sum())
    if not np.isfinite(tot) or tot <= eps:
        return 1.0
    p = v / tot
    H = -float(np.sum(p * np.log(p + eps)))
    return float(H / np.log(len(p)))


# ============================================================
# Plot: 3 rows x len(sigmas) cols
# ============================================================
def plot_grid(dataset: str, variants: dict, entropies: dict, idxs: np.ndarray, outpath: Path):
    sigmas_sorted = sorted(variants.keys())
    ncols = len(sigmas_sorted)
    nrows = len(idxs)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        constrained_layout=True
    )

    # axes handling if nrows/ncols = 1
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # titles: one per sigma
    for j, s in enumerate(sigmas_sorted):
        axes[0, j].set_title(f"gauss σ={s:.2f}\nH={entropies[s]:.3f}", fontsize=11)

    # images
    for i, idx in enumerate(idxs):
        for j, s in enumerate(sigmas_sorted):
            ax = axes[i, j]
            ax.imshow(variants[s][idx], cmap="gray")
            ax.axis("off")

    fig.suptitle(f"{dataset} — gauss noise levels (32×32, grayscale)", fontsize=13)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def plot_multi_dataset_grid(
    datasets: list[str],
    sigmas: list[float],
    variants_all: dict[str, dict[float, np.ndarray]],   # ds -> (sigma -> X (N,H,W))
    entropies_all: dict[str, dict[float, float]],       # ds -> (sigma -> H)
    idx_per_ds: dict[str, int],
    outpath: Path,
):
    sigmas_sorted = sorted(sigmas)
    nrows = len(datasets)
    ncols = len(sigmas_sorted)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        constrained_layout=True
    )

    # Handle axes shape if nrows/ncols = 1
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # Column titles (sigmas)
    for j, s in enumerate(sigmas_sorted):
        axes[0, j].set_title(f"gauss σ={s:.2f}", fontsize=11)

    # Fill cells
    for i, ds in enumerate(datasets):
        idx = idx_per_ds[ds]

        # Row label with entropies (compact)
        ent_str = " | ".join([f"H{ss:.2f}={entropies_all[ds][ss]:.3f}" for ss in sigmas_sorted])
        # Put as y-label on first column
        axes[i, 0].set_ylabel(f"{ds}\n{ent_str}", fontsize=10)

        for j, s in enumerate(sigmas_sorted):
            ax = axes[i, j]
            ax.imshow(variants_all[ds][s][idx], cmap="gray")
            ax.axis("off")

    fig.suptitle("Gauss noise levels (32×32, grayscale) — one sample per dataset", fontsize=13)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="entropy_gauss_viz_32x32")
    ap.add_argument("--datasets", type=str,
                    default="usps,optdigits,pathmnist,dermamnist,organcmnist,kmnist")
    ap.add_argument("--sigmas", type=str, default="0.00,0.10,0.20")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_hw", type=str, default="32,32")
    ap.add_argument("--n_show", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    sigmas = [round(s, 2) for s in sigmas]

    H_out, W_out = [int(x.strip()) for x in args.out_hw.split(",")]
    out_hw = (H_out, W_out)

    rng = np.random.default_rng(args.seed)

    # Fuerza 4 datasets (como en tu script actual)
    datasets = ["digits", "olivetti", "cifar10", "pneumoniamnist"]

    variants_all = {}   # ds -> {sigma -> X}
    entropies_all = {}  # ds -> {sigma -> H}
    idx_per_ds = {}     # ds -> idx (un solo ejemplo)

    for ds in datasets:
        print(f"\nDataset: {ds}")

        variants = {}
        entropies = {}

        for s in sigmas:
            X_flat, y, (H, W) = load_dataset(
                ds,
                n_samples=150,
                seed=123,
                variant="gauss",
                noise_sigma=s,
                out_hw=out_hw
            )

            X = X_flat.reshape(len(X_flat), H, W).astype(np.float32, copy=False)

            variants[s] = X
            Pm = power_spectrum_mean(X)
            entropies[s] = spectral_entropy_norm(Pm)

        # elige 1 índice por dataset (mismo idx para todas las sigmas de ese dataset)
        N = len(next(iter(variants.values())))
        idx = int(rng.integers(0, N)) if N > 0 else 0
        idx_per_ds[ds] = idx

        variants_all[ds] = variants
        entropies_all[ds] = entropies

        ent_str = " | ".join([f"H(σ={s:.2f})={entropies[s]:.4f}" for s in sorted(entropies.keys())])
        print(ent_str)

    # UNA sola figura con 4 datasets × 3 sigmas
    outpath = out_dir / "F_entropy_grid_gauss_4datasets.png"
    plot_multi_dataset_grid(
        datasets=datasets,
        sigmas=sigmas,
        variants_all=variants_all,
        entropies_all=entropies_all,
        idx_per_ds=idx_per_ds,
        outpath=outpath,
    )

    print(f"\n[ok] saved → {outpath}")
    print(f"Done. Output in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
