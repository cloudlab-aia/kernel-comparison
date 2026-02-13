#src/classical/preprocessing.py
import numpy as np
import numpy as np

def fft_1d_ab_encoding_lowfreq(
    X_flat,
    shape,
    k=4,
    lam=np.pi,
    return_norm=True,
):
    """
    FFT-1D a+ib encoding with LOW-FREQUENCY truncation.

    Parameters
    ----------
    X_flat : (N, d)
    shape  : must be (d,)
    k      : number of low-frequency bins (excluding DC)
    lam    : phase mixing strength
    """

    X_flat = np.asarray(X_flat, float)
    N, d = X_flat.shape

    if shape != (d,):
        raise ValueError(f"fft_1d_ab_encoding_lowfreq expects shape=(d,), got {shape}")

    # -----------------------------------------
    # Center signals
    # -----------------------------------------
    X = X_flat - X_flat.mean(axis=1, keepdims=True)

    # -----------------------------------------
    # Real FFT
    # -----------------------------------------
    F = np.fft.rfft(X, axis=1)   # (N, d//2+1)

    # -----------------------------------------
    # Remove DC, keep lowest frequencies
    # -----------------------------------------
    F_sel = F[:, 1:k+1]           # (N, k)

    # -----------------------------------------
    # Magnitude & phase
    # -----------------------------------------
    A = np.abs(F_sel)
    theta = np.angle(F_sel)

    # -----------------------------------------
    # Global spectral norm
    # -----------------------------------------
    r = np.linalg.norm(A, axis=1)

    # -----------------------------------------
    # Log-spectral weighting
    # -----------------------------------------
    w = np.log1p(A)
    cap = np.quantile(w, 0.95)
    w = np.clip(w, 0.0, cap) / (cap + 1e-12)

    # -----------------------------------------
    # Phase mixing
    # -----------------------------------------
    phi = theta + lam * w
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    # -----------------------------------------
    # Unit-modulus complex embedding
    # -----------------------------------------
    Z = np.cos(phi) + 1j * np.sin(phi)

    if return_norm:
        return Z, r
    return Z


"""
def fft_nd_ab_encoding_radial(
    X_flat,
    shape,
    k=4,
    lam=np.pi,
    return_norm=True,
):
    """"""    
    FT-ND based a+ib encoding with RADIAL low-frequency truncation.

    - ND generalization of fft2_image_ab_encoding_radial
    - fftshifted (true radial symmetry)
    - DC removal
    - robust log-spectral weighting

    X_flat: (N, prod(shape))
    shape: tuple, e.g. (H, W), (H, W, D), ...""""""

    X_flat = np.asarray(X_flat, float)
    N, d = X_flat.shape

    if np.prod(shape) != d:
        raise ValueError(
            f"shape {shape} incompatible with d={d}"
        )

    ndim = len(shape)

    # -------------------------------------------------
    # Reshape + center per sample
    # -------------------------------------------------
    X = X_flat.reshape(N, *shape)
    X = X - X.mean(axis=tuple(range(1, ndim + 1)), keepdims=True)

    # -------------------------------------------------
    # FFT-ND + shift
    # -------------------------------------------------
    F = np.fft.fftshift(
        np.fft.fftn(X, axes=tuple(range(1, ndim + 1))),
        axes=tuple(range(1, ndim + 1)),
    )

    # -------------------------------------------------
    # Build ND radial mask
    # -------------------------------------------------
    grids = np.meshgrid(
        *[np.arange(-s // 2, s // 2) for s in shape],
        indexing="ij",
    )
    R = np.sqrt(sum(g**2 for g in grids))
    mask = R <= k

    # Remove DC (center index)
    center = tuple(s // 2 for s in shape)
    mask[center] = False

    idx = np.where(mask)

    # -------------------------------------------------
    # Select coefficients
    # -------------------------------------------------
    F_sel = F[(slice(None),) + idx]  # (N, m)

    # -------------------------------------------------"""  """
    # Magnitude & phase
    # -------------------------------------------------
    A = np.abs(F_sel)
    theta = np.angle(F_sel)

    # -------------------------------------------------
    # Global spectral norm
    # -------------------------------------------------
    r = np.linalg.norm(A, axis=1)

    # -------------------------------------------------
    # Log-spectral weights + clipping
    # -------------------------------------------------
    w = np.log1p(A)
    cap = np.quantile(w, 0.95)
    w = np.clip(w, 0.0, cap) / (cap + 1e-12)

    # -------------------------------------------------
    # Phase mixing
    # -------------------------------------------------
    phi = theta + lam * w
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    # -------------------------------------------------
    # Unit-modulus complex embedding
    # -------------------------------------------------
    Z = np.cos(phi) + 1j * np.sin(phi)

    if return_norm:
        return Z, r
    return Z
"""

def fft_nd_ab_encoding_radial(
    X_flat,
    shape,
    k=4,
    lam=np.pi,
    return_norm=True,
):
    """
    FFT-ND based a+ib encoding with RADIAL low-frequency truncation.

    Small upgrades for better practical performance (minimal-invasion):
      1) Per-sample clipping/normalization of log-magnitude weights (more robust)
      2) Allow a small lambda grid via `lam` (float OR iterable of floats)
         - If iterable, returns embeddings for each lambda.
      3) k can be absolute (int) OR relative (float in (0,1]) interpreted as fraction of Nyquist radius.

    Parameters
    ----------
    X_flat : array-like, shape (N, prod(shape))
    shape  : tuple[int, ...]
    k      : int OR float
             - int: radial cutoff in frequency bins (as before)
             - float: fraction of Nyquist radius (0<k<=1). k_frac=0.25 means use 25% of Nyquist.
    lam    : float OR iterable[float]
             - float: single lambda
             - iterable: grid of lambdas, e.g. [0, pi/4, pi/2, 3pi/4, pi]
    return_norm : bool
        If True, also returns r = ||A||_2 (per sample) for optional downstream use.

    Returns
    -------
    If lam is a float:
        Z : (N, m) complex unit-modulus
        r : (N,) if return_norm else not returned
    If lam is an iterable:
        Zs : (L, N, m) complex, one embedding per lambda in given order
        r  : (N,) if return_norm else not returned
    """

    X_flat = np.asarray(X_flat, float)
    N, d = X_flat.shape

    if np.prod(shape) != d:
        raise ValueError(f"shape {shape} incompatible with d={d}")

    ndim = len(shape)

    # -------------------------------------------------
    # k: absolute bins OR relative fraction of Nyquist
    # -------------------------------------------------
    if isinstance(k, (float, np.floating)) and not isinstance(k, (bool, np.bool_)):
        k_frac = float(k)
        if not (0.0 < k_frac <= 1.0):
            raise ValueError("If k is float, it must be a fraction in (0,1].")
        # Nyquist radius in shifted grid is min(shape)//2
        nyq = min(int(s) for s in shape) // 2
        k_eff = max(1, int(round(k_frac * nyq)))
    else:
        k_eff = int(k)
        if k_eff < 1:
            raise ValueError("k must be >= 1 (or a fraction in (0,1])")

    # -------------------------------------------------
    # Reshape + center per sample (DC removal)
    # -------------------------------------------------
    X = X_flat.reshape(N, *shape)
    X = X - X.mean(axis=tuple(range(1, ndim + 1)), keepdims=True)

    # -------------------------------------------------
    # FFT-ND + shift
    # -------------------------------------------------
    F = np.fft.fftshift(
        np.fft.fftn(X, axes=tuple(range(1, ndim + 1))),
        axes=tuple(range(1, ndim + 1)),
    )

    # -------------------------------------------------
    # Build ND radial mask
    # -------------------------------------------------
    grids = np.meshgrid(
        *[np.arange(-s // 2, s // 2) for s in shape],
        indexing="ij",
    )
    R = np.sqrt(sum(g ** 2 for g in grids))
    mask = (R <= k_eff)

    # Remove DC (center index)
    center = tuple(s // 2 for s in shape)
    mask[center] = False

    idx = np.where(mask)

    # -------------------------------------------------
    # Select coefficients
    # -------------------------------------------------
    F_sel = F[(slice(None),) + idx]  # (N, m)

    # -------------------------------------------------
    # Magnitude & phase
    # -------------------------------------------------
    A = np.abs(F_sel)               # (N, m)
    theta = np.angle(F_sel)         # (N, m)

    # -------------------------------------------------
    # Global spectral norm (optional diagnostic/weight)
    # -------------------------------------------------
    r = np.linalg.norm(A, axis=1)

    # -------------------------------------------------
    # Log-spectral weights + *per-sample* clipping
    # -------------------------------------------------
    w = np.log1p(A)  # (N, m)

    # cap per sample (robust): q=0.95 along m dimension
    cap = np.quantile(w, 0.95, axis=1, keepdims=True)
    w = np.clip(w, 0.0, cap) / (cap + 1e-12)  # now in ~[0,1] per sample

    # -------------------------------------------------
    # Lambda: single value or small grid
    # -------------------------------------------------
    if isinstance(lam, (list, tuple, np.ndarray)):
        lam_grid = [float(x) for x in lam]
        if len(lam_grid) == 0:
            raise ValueError("lam iterable is empty. Provide at least one value.")
    else:
        lam_grid = None
        lam = float(lam)

    # -------------------------------------------------
    # Phase mixing and unit-modulus embedding
    # -------------------------------------------------
    def _embed_for_lambda(lmb: float):
        phi = theta + lmb * w
        # wrap to [-pi, pi] for numerical stability (optional)
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        return np.cos(phi) + 1j * np.sin(phi)

    if lam_grid is None:
        Z = _embed_for_lambda(lam)
        if return_norm:
            return Z, r
        return Z

    # Return embeddings for each lambda in grid: (L, N, m)
    Zs = np.stack([_embed_for_lambda(lmb) for lmb in lam_grid], axis=0)

    if return_norm:
        return Zs, r
    return Zs




