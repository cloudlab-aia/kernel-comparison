# src/classical/common_eval.py
import os, csv, time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
def complex_linear_kernel(Za: np.ndarray, Zb: np.ndarray) -> np.ndarray:
    """
    Linear kernel for complex features using Hermitian inner product, then take Re(.)
    K_ij = Re( <Za_i, Zb_j> ) = Re( Za_i @ conj(Zb_j) )
    """
    return np.real(Za @ np.conjugate(Zb).T)


def complex_to_real_features(Z: np.ndarray) -> np.ndarray:
    """
    Map complex features to real by concatenating real and imag parts: [Re(Z), Im(Z)].
    Shape: (N, m) complex -> (N, 2m) real
    """
    return np.concatenate([np.real(Z), np.imag(Z)], axis=1).astype(np.float64, copy=False)

def center_kernel(K: np.ndarray) -> np.ndarray:
    N = K.shape[0]
    one = np.ones((N, N), dtype=K.dtype) / N
    return K - one @ K - K @ one + one @ K @ one


def kernel_alignment(K: np.ndarray, Ky: np.ndarray) -> float:
    num = float(np.sum(K * Ky))
    den = float(np.linalg.norm(K) * np.linalg.norm(Ky))
    return float(num / (den + 1e-12))


def stats_within_between_multi(K: np.ndarray, y: np.ndarray):
    y = np.asarray(y)
    N = len(y)
    iu = np.triu_indices(N, k=1)
    same = (y[:, None] == y[None, :])[iu]
    vals = K[iu]
    mu_within = float(np.mean(vals[same])) if np.any(same) else 0.0
    mu_between = float(np.mean(vals[~same])) if np.any(~same) else 0.0
    return mu_within, mu_between, mu_within - mu_between

def append_row_csv(path: str, row: dict, fieldnames: list[str]):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def eval_splits_precomputed_kernel(
    *,
    X_flat: np.ndarray,
    y: np.ndarray,
    shape: tuple[int, int],
    k_rad: int,
    m_k: int,
    shots: int,
    seeds=[1,2,3,4,5],
    lam=np.pi,
    build_kernel_fn=None,          # (Ztr, ntr, shots)->(Ktr, aux)
    build_cross_kernel_fn=None,    # (Zte, nte, Ztr, ntr, shots)->(Kte_tr, aux)
    dft_embed_fn=None,             # fft_nd_ab_encoding_radial
    pca_split_fn=None,             # pca_fit_transform_split
    rp_split_fn=None,              # rp_transform_split
    method: str = "DFT",
):
    accs, f1s = [], []
    aligns, deltas = [], []

    idx_all = np.arange(len(y))

    for seed in seeds:
        idx_tr, idx_te = train_test_split(
            idx_all, test_size=0.3, stratify=y, random_state=seed
        )

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_flat[idx_tr])
        Xte = scaler.transform(X_flat[idx_te])

        ytr = y[idx_tr]
        yte = y[idx_te]

        if method == "DFT":
            Ztr, ntr = dft_embed_fn(Xtr, shape=shape, k=k_rad, lam=lam, return_norm=True)
            Zte, nte = dft_embed_fn(Xte, shape=shape, k=k_rad, lam=lam, return_norm=True)

        elif method == "PCA":
            (Ztr, ntr), (Zte, nte) = pca_split_fn(Xtr, Xte, m=m_k, seed=seed, lam=lam)

        elif method == "RP":
            (Ztr, ntr), (Zte, nte) = rp_split_fn(Xtr, Xte, m=m_k, seed=seed, lam=lam)

        else:
            raise ValueError("method must be DFT|PCA|RP")

        # --- Train kernel (NxN)
        K_tr = build_kernel_fn(Ztr, ntr, shots=shots, seed=seed)

        # --- Cross kernel (Nte x Ntr)
        K_te_tr, _ = build_cross_kernel_fn(Zte, nte, Ztr, ntr, shots=shots, seed=seed)

        # --- Alignment/delta on TRAIN kernel only
        Ky_tr = np.where(ytr[:, None] == ytr[None, :], 1.0, -1.0)
        Kc_tr  = center_kernel(K_tr)
        Kyc_tr = center_kernel(Ky_tr)
        aligns.append(kernel_alignment(Kc_tr, Kyc_tr))
        _, _, dlt = stats_within_between_multi(Kc_tr, ytr)
        deltas.append(dlt)

        # --- SVM
        clf = SVC(kernel="precomputed")
        clf.fit(K_tr, ytr)
        y_pred = clf.predict(K_te_tr)

        accs.append(accuracy_score(yte, y_pred))
        f1s.append(f1_score(yte, y_pred, average="macro"))

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "alignment_mean": float(np.mean(aligns)),
        "alignment_std": float(np.std(aligns)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
    }



def median_heuristic_gamma(X: np.ndarray, max_pairs: int = 4000, seed: int = 0) -> float:
    """
    Median heuristic for RBF: gamma = 1 / (2 * median(||xi-xj||^2))
    Uses a subsample of pairs to keep it cheap.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 2:
        return 1.0

    # sample indices
    m = min(n, int(np.sqrt(max_pairs)) + 1)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X[idx]

    # pairwise squared distances (upper triangle)
    G = Xs @ Xs.T
    sq = np.clip(np.diag(G)[:, None] + np.diag(G)[None, :] - 2.0 * G, 0.0, None)
    iu = np.triu_indices(m, k=1)
    vals = sq[iu]
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    return float(1.0 / (2.0 * med + 1e-12))


def eval_splits_classical_svm_matched(
    *,
    X_flat: np.ndarray,
    y: np.ndarray,
    shape: tuple[int, int] | tuple[int, ...],
    k_rad: int,
    m_k: int,
    seeds=[1,2,3,4,5],
    lam=np.pi,
    dft_embed_fn=None,
    pca_split_fn=None,
    rp_split_fn=None,
    method: str = "DFT",     # DFT|PCA|RP
    baseline: str = "linear" # linear|rbf
):
    """
    Classical baselines matched to the SAME embeddings (DFT/PCA/RP) and SAME splits/scaling.
    Trains a standard SVC on a classical kernel computed from the embedded features Z (not precomputed quantum kernel).
    """
    accs, f1s = [], []
    aligns, deltas = [], []

    idx_all = np.arange(len(y))

    for seed in seeds:
        idx_tr, idx_te = train_test_split(
            idx_all, test_size=0.3, stratify=y, random_state=seed
        )

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_flat[idx_tr])
        Xte = scaler.transform(X_flat[idx_te])

        ytr = y[idx_tr]
        yte = y[idx_te]

        if method == "DFT":
            Ztr, _ = dft_embed_fn(Xtr, shape=shape, k=k_rad, lam=lam, return_norm=True)
            Zte, _ = dft_embed_fn(Xte, shape=shape, k=k_rad, lam=lam, return_norm=True)
        elif method == "PCA":
            (Ztr, _), (Zte, _) = pca_split_fn(Xtr, Xte, m=m_k, seed=seed, lam=lam)
        elif method == "RP":
            (Ztr, _), (Zte, _) = rp_split_fn(Xtr, Xte, m=m_k, seed=seed, lam=lam)
        else:
            raise ValueError("method must be DFT|PCA|RP")

        is_cplx = np.iscomplexobj(Ztr) or np.iscomplexobj(Zte)

        if baseline == "linear":
            if is_cplx:
                K_tr = complex_linear_kernel(Ztr, Ztr)
                K_te_tr = complex_linear_kernel(Zte, Ztr)
            else:
                K_tr = Ztr @ Ztr.T
                K_te_tr = Zte @ Ztr.T

        elif baseline == "rbf":
            # RBF over real space; if complex, lift to R^(2m)
            if is_cplx:
                Ztr_r = complex_to_real_features(Ztr)
                Zte_r = complex_to_real_features(Zte)
            else:
                Ztr_r = Ztr
                Zte_r = Zte

            gamma = median_heuristic_gamma(Ztr_r, seed=seed)
            # implement RBF manually (avoid sklearn checks)
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
            Xn = np.sum(Ztr_r * Ztr_r, axis=1)
            Yn = np.sum(Zte_r * Zte_r, axis=1)

            K_tr = np.exp(-gamma * (Xn[:, None] + Xn[None, :] - 2.0 * (Ztr_r @ Ztr_r.T)))
            K_te_tr = np.exp(-gamma * (Yn[:, None] + Xn[None, :] - 2.0 * (Zte_r @ Ztr_r.T)))

        else:
            raise ValueError("baseline must be linear|rbf")


        # --- Alignment/delta on TRAIN kernel only (same diagnostics)
        Ky_tr = np.where(ytr[:, None] == ytr[None, :], 1.0, -1.0)
        Kc_tr  = center_kernel(K_tr)
        Kyc_tr = center_kernel(Ky_tr)
        aligns.append(kernel_alignment(Kc_tr, Kyc_tr))
        _, _, dlt = stats_within_between_multi(Kc_tr, ytr)
        deltas.append(dlt)

        # --- SVM with precomputed classical kernel
        clf = SVC(kernel="precomputed")
        clf.fit(K_tr, ytr)
        y_pred = clf.predict(K_te_tr)

        accs.append(accuracy_score(yte, y_pred))
        f1s.append(f1_score(yte, y_pred, average="macro"))

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "alignment_mean": float(np.mean(aligns)),
        "alignment_std": float(np.std(aligns)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
    }

