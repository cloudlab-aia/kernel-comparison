# experiments/run_real2d_matched.py
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.classical.preprocessing import fft_1d_ab_encoding_lowfreq, fft_nd_ab_encoding_radial

from src.quantum.pes_kernel import build_gram_matrix as build_K_dg
from src.quantum.pes_kernel import build_cross_gram_matrix as build_cross_K_dg

from src.classical.baselines_matched import pca_fit_transform_split, rp_transform_split
from src.classical.nd_matched import run_matched_nd
from src.classical.dataset_factory import load_dataset


SEED_DATASET = 123
SEEDS_EXPERIMENT = [1,2,3,4,5]

def norm_ds_name(ds: str) -> str:
    ds = str(ds).strip()
    return ds

def csv_name_for_ds(outdir: str, ds: str, val=None) -> str:
    ds_norm = norm_ds_name(ds)
    safe = ds_norm.replace("/", "_")

    if val is not None:
        return os.path.join(outdir, f"rp_real_2d_color_{safe}_{val:.3f}.csv")
    else:
        return os.path.join(outdir, f"rp_real_2d_color_{safe}.csv")

        

def main():
    SHOTS = 512
    N_SAMPLES = 150
    N_QUBITS_LIST = [2,3,4,5,6]
    OUTDIR = "results_dft_vs_pca_vs_rp_real_2D"
    os.makedirs(OUTDIR, exist_ok=True)

    FIELDNAMES = [
        "dataset", "preproc",
        "n_qubits", "budget", "k", "m_used",
        "alignment", "mu_within", "mu_between", "delta",
        "accuracy_mean", "accuracy_std", "f1_mean", "f1_std",
        "time_kernel", "samples", "shots", "seed","model"
    ]

    DATASETS_2D = ["breastmnist", "cifar10", "dermamnist", "digits", "dtd", "emnist/byclass", "fashion_mnist", "fer2013",
                    "geometric_shapes", "gtsrb", "kmnist", "octmnist", "olivetti", "omniglot", "optdigits", "pneumoniamnist",
                    "retinamnist", "stl10", "svhn_cropped", "usps",]
    for ds in DATASETS_2D:
        ds = norm_ds_name(ds)
        for noise_sigma in [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
            out_csv = csv_name_for_ds(OUTDIR, ds, val=noise_sigma)
            X_flat, y, shape = load_dataset(
                ds,
                n_samples=N_SAMPLES,
                seed=SEED_DATASET,
                variant="gauss",
                noise_sigma=noise_sigma
            )
            print(f"\nDataset: {ds}_gauss | shape={shape} | X={X_flat.shape} | classes={len(np.unique(y))}")
            if len(shape) == 1:
                dft_embed_fn = fft_1d_ab_encoding_lowfreq
            else:
                dft_embed_fn = fft_nd_ab_encoding_radial
            run_matched_nd(
                dataset_name=f"{ds}_real",
                out_csv=out_csv,
                X_flat=X_flat,
                y=y,
                shape=shape,
                n_qubits_list=N_QUBITS_LIST,
                shots=SHOTS,
                n_samples=N_SAMPLES,
                fieldnames=FIELDNAMES,
                m_min=2,
                build_kernel_fn=build_K_dg,
                build_cross_kernel_fn=build_cross_K_dg,
                fft_nd_ab_encoding_radial_fn=dft_embed_fn,
                pca_split_fn=pca_fit_transform_split,
                rp_split_fn=rp_transform_split,
                seeds=SEEDS_EXPERIMENT
            )


if __name__ == "__main__":
    main()
