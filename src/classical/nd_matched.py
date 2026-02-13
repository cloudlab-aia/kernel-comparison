# src/classical/nd_matched.py
import time
import numpy as np

from .common_eval import append_row_csv, eval_splits_precomputed_kernel, eval_splits_classical_svm_matched


def number_of_selected_freqs(shape, k: int) -> int:
    """
    Returns effective feature dimension m_used for a given selector parameter k.

    - 1D: k = number of low-frequency bins (excluding DC) -> m_used = k
    - ND: k = radial cutoff -> m_used = #freqs in radial mask (excluding DC)
    """
    if len(shape) == 1:
        return int(k)

    grids = np.meshgrid(
        *[np.arange(-s // 2, s // 2) for s in shape],
        indexing="ij",
    )
    R = np.sqrt(sum(g**2 for g in grids))
    mask = (R <= k)
    center = tuple(s // 2 for s in shape)
    mask[center] = False
    return int(np.sum(mask))



def iter_k_blocks_general(n_qubits_list, shape, m_min=2):
    """
    Yields (n_qubits, budget, ks)

    - 1D: k = 2^n (clipped)
    - ND: bucket by the intrinsic range (2^(n-1), 2^n], independent of list contents.
    """
    max_budget = 2 ** max(n_qubits_list)

    # ----------------------------
    # 1D: only powers of two
    # ----------------------------
    if len(shape) == 1:
        d = int(shape[0])
        k_max = min(d // 2, max_budget)
        for n in n_qubits_list:
            budget = 2 ** n
            k = min(budget, k_max)
            if k >= m_min:
                yield n, budget, [k]
        return

    # ----------------------------
    # ND: (2^(n-1), 2^n]
    # ----------------------------
    k_max = min(shape) // 2

    for n in n_qubits_list:
        budget = 2 ** n
        prev_budget = 2 ** (n - 1)  # <-- CLAVE: depende de n, no de la lista

        ks = []
        for k in range(1, k_max + 1):
            m_k = number_of_selected_freqs(shape, k)
            if m_k < m_min:
                continue
            if m_k > budget:
                break
            if prev_budget < m_k <= budget:
                ks.append(k)

        if ks:
            yield n, budget, ks
        
def run_matched_nd(
    *,
    dataset_name: str,
    out_csv: str,
    X_flat: np.ndarray,     # (N, prod(shape))
    y: np.ndarray,
    shape: tuple[int, ...],
    n_qubits_list,
    shots: int,
    n_samples: int,
    fieldnames,
    m_min: int,
    # --- kernel fns (must be split-fair)
    build_kernel_fn,               # (Ztr, ntr, shots)->(Ktr, aux)
    build_cross_kernel_fn,         # (Zte, nte, Ztr, ntr, shots)->(Kte_tr, aux)
    # --- embeddings
    fft_nd_ab_encoding_radial_fn,  # DFT embed (X, shape, k, lam, return_norm)->(Z,norms)
    pca_split_fn,                  # (Xtr, Xte, m, seed, lam)->((Ztr,ntr),(Zte,nte))
    rp_split_fn,                   # same
    seeds=[1,2,3,4,5],
    lam=np.pi,
):
    """
    FAIR evaluation:
      - For each (n_qubits, k_rad), evaluate DFT/PCA/RP by:
          * fitting StandardScaler on TRAIN only
          * fitting PCA on TRAIN only (for PCA)
          * using same RP matrix for train/test (for RP)
          * building K_train and K_test_train (cross kernel)
          * training SVM and evaluating on test
      - Aggregate metrics across seeds and write one row per method per k_rad.
    """
    k_max = min(shape) // 2

    print(f"\n===== {dataset_name} | ND MATCHED: DFT_ND_RADIAL vs PCA_MATCHED vs RP_MATCHED =====")
    print(f"shape={shape}, k_max={k_max}\n")

    for n, budget, ks in iter_k_blocks_general(n_qubits_list, shape, m_min=m_min):
        print("\n" + "-" * 80)
        print(f"[ND] n_qubits={n} | budget={budget} | radii ks: {ks[0]}..{ks[-1]} (count={len(ks)})")
        print("-" * 80)

        for k_sel in ks:
            # Effective dimension for this radial cutoff (no need to embed globally)
            m_k = number_of_selected_freqs(shape, k_sel)
            if m_k < m_min:
                continue

            # ----------------------------------------------------------
            # DFT
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            stats = eval_splits_precomputed_kernel(
                X_flat=X_flat,
                y=y,
                shape=shape,
                k_rad=int(k_sel),
                m_k=int(m_k),
                shots=shots,
                seeds=seeds,
                lam=lam,
                build_kernel_fn=build_kernel_fn,
                build_cross_kernel_fn=build_cross_kernel_fn,
                dft_embed_fn=fft_nd_ab_encoding_radial_fn,
                pca_split_fn=pca_split_fn,
                rp_split_fn=rp_split_fn,
                method="DFT",
            )
            t_kernel = time.perf_counter() - t0

            row = {
                "dataset": dataset_name,
                "preproc": "DFT_ND_RADIAL",
                "n_qubits": int(n),
                "budget": int(budget),
                "k": int(k_sel),
                "m_used": int(m_k),
                "alignment": float(stats["alignment_mean"]),
                "mu_within": 0.0,   # optional: add if you compute these per split
                "mu_between": 0.0,  # optional
                "delta": float(stats["delta_mean"]),
                "accuracy_mean": float(stats["acc_mean"]),
                "accuracy_std": float(stats["acc_std"]),
                "f1_mean": float(stats["f1_mean"]),
                "f1_std": float(stats["f1_std"]),
                "time_kernel": float(t_kernel),
                "samples": int(n_samples),
                "shots": int(shots),
            }
            append_row_csv(out_csv, row, fieldnames)

            print(
                f"[DFT_ND_RADIAL] n={n:2d} k={k_sel:3d} m={m_k:4d} | "
                f"align={row['alignment']:.3f} Δ={row['delta']:.3f} | "
                f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f} | "
                f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f} | "
                f"time={t_kernel:6.2f}s"
            )

            # ----------------------------------------------------------
            # PCA
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            stats = eval_splits_precomputed_kernel(
                X_flat=X_flat,
                y=y,
                shape=shape,
                k_rad=int(k_sel),
                m_k=int(m_k),
                shots=shots,
                seeds=seeds,
                lam=lam,
                build_kernel_fn=build_kernel_fn,
                build_cross_kernel_fn=build_cross_kernel_fn,
                dft_embed_fn=fft_nd_ab_encoding_radial_fn,
                pca_split_fn=pca_split_fn,
                rp_split_fn=rp_split_fn,
                method="PCA",
            )
            t_kernel = time.perf_counter() - t0

            row = {
                "dataset": dataset_name,
                "preproc": "PCA_MATCHED",
                "n_qubits": int(n),
                "budget": int(budget),
                "k": int(k_sel),
                "m_used": int(m_k),
                "alignment": float(stats["alignment_mean"]),
                "mu_within": 0.0,
                "mu_between": 0.0,
                "delta": float(stats["delta_mean"]),
                "accuracy_mean": float(stats["acc_mean"]),
                "accuracy_std": float(stats["acc_std"]),
                "f1_mean": float(stats["f1_mean"]),
                "f1_std": float(stats["f1_std"]),
                "time_kernel": float(t_kernel),
                "samples": int(n_samples),
                "shots": int(shots),
            }
            append_row_csv(out_csv, row, fieldnames)

            print(
                f"[PCA_MATCHED]   n={n:2d} k={k_sel:3d} m={m_k:4d} | "
                f"align={row['alignment']:.3f} Δ={row['delta']:.3f} | "
                f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f} | "
                f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f} | "
                f"time={t_kernel:6.2f}s"
            )

            # ----------------------------------------------------------
            # RP
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            stats = eval_splits_precomputed_kernel(
                X_flat=X_flat,
                y=y,
                shape=shape,
                k_rad=int(k_sel),
                m_k=int(m_k),
                shots=shots,
                seeds=seeds,
                lam=lam,
                build_kernel_fn=build_kernel_fn,
                build_cross_kernel_fn=build_cross_kernel_fn,
                dft_embed_fn=fft_nd_ab_encoding_radial_fn,
                pca_split_fn=pca_split_fn,
                rp_split_fn=rp_split_fn,
                method="RP",
            )
            t_kernel = time.perf_counter() - t0

            row = {
                "dataset": dataset_name,
                "preproc": "RP_MATCHED",
                "n_qubits": int(n),
                "budget": int(budget),
                "k": int(k_sel),
                "m_used": int(m_k),
                "alignment": float(stats["alignment_mean"]),
                "mu_within": 0.0,
                "mu_between": 0.0,
                "delta": float(stats["delta_mean"]),
                "accuracy_mean": float(stats["acc_mean"]),
                "accuracy_std": float(stats["acc_std"]),
                "f1_mean": float(stats["f1_mean"]),
                "f1_std": float(stats["f1_std"]),
                "time_kernel": float(t_kernel),
                "samples": int(n_samples),
                "shots": int(shots),
            }
            append_row_csv(out_csv, row, fieldnames)

            print(
                f"[RP_MATCHED]    n={n:2d} k={k_sel:3d} m={m_k:4d} | "
                f"align={row['alignment']:.3f} Δ={row['delta']:.3f} | "
                f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f} | "
                f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f} | "
                f"time={t_kernel:6.2f}s"
            )

            # ---------------------------
            # CLASSICAL BASELINE: DFT + LINEAR
            # ---------------------------
            t0 = time.perf_counter()
            stats = eval_splits_classical_svm_matched(
                X_flat=X_flat, y=y, shape=shape,
                k_rad=int(k_sel), m_k=int(m_k),
                seeds=seeds, lam=lam,
                dft_embed_fn=fft_nd_ab_encoding_radial_fn,
                pca_split_fn=pca_split_fn,
                rp_split_fn=rp_split_fn,
                method="DFT",
                baseline="linear",
            )
            t_bl = time.perf_counter() - t0

            row = {
                "dataset": dataset_name,
                "preproc": "DFT_ND_RADIAL",
                "model": "SVM_LINEAR",          # <-- añade esta columna al CSV
                "n_qubits": int(n),
                "budget": int(budget),
                "k": int(k_sel),
                "m_used": int(m_k),
                "alignment": float(stats["alignment_mean"]),
                "delta": float(stats["delta_mean"]),
                "accuracy_mean": float(stats["acc_mean"]),
                "accuracy_std": float(stats["acc_std"]),
                "f1_mean": float(stats["f1_mean"]),
                "f1_std": float(stats["f1_std"]),
                "time_kernel": float(t_bl),
                "samples": int(n_samples),
                "shots": 0,                     # no aplica
            }
            append_row_csv(out_csv, row, fieldnames)

            # ---------------------------
            # CLASSICAL BASELINE: DFT + RBF
            # ---------------------------
            t0 = time.perf_counter()
            stats = eval_splits_classical_svm_matched(
                X_flat=X_flat, y=y, shape=shape,
                k_rad=int(k_sel), m_k=int(m_k),
                seeds=seeds, lam=lam,
                dft_embed_fn=fft_nd_ab_encoding_radial_fn,
                pca_split_fn=pca_split_fn,
                rp_split_fn=rp_split_fn,
                method="DFT",
                baseline="rbf",
            )
            t_bl = time.perf_counter() - t0

            row = {
                "dataset": dataset_name,
                "preproc": "DFT_ND_RADIAL",
                "model": "SVM_RBF",
                "n_qubits": int(n),
                "budget": int(budget),
                "k": int(k_sel),
                "m_used": int(m_k),
                "alignment": float(stats["alignment_mean"]),
                "delta": float(stats["delta_mean"]),
                "accuracy_mean": float(stats["acc_mean"]),
                "accuracy_std": float(stats["acc_std"]),
                "f1_mean": float(stats["f1_mean"]),
                "f1_std": float(stats["f1_std"]),
                "time_kernel": float(t_bl),
                "samples": int(n_samples),
                "shots": 0,
            }
            append_row_csv(out_csv, row, fieldnames)
