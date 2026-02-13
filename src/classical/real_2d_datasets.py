# src/classical/real_2d_datasets.py
from __future__ import annotations

import numpy as np
from typing import Dict, List


def load_real_2d_dataset(
    name: str,
    n_samples: int | None = None,
    classes: list[int] | None = None,
    seed: int = 123,
    grayscale: bool = True,
    out_hw: tuple[int, int] | None = None,   # None = keep original resolution
    variant: str = "raw",                    # 'raw' | 'gauss'
    noise_sigma: float = 0.0,                # std of additive Gaussian noise in [0,1]
    labels_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Returns:
      X_flat: (N, H*W) float32 in [0,1]
      y:      (N,) int
      shape:  (H, W)

    Policy (homogeneous):
      - load raw arrays
      - optional class filtering
      - normalize to [0,1]
      - convert to grayscale -> (N,H,W)
      - optional resize (if out_hw is not None)
      - strict stratified subsample to n_samples
      - apply variant AFTER subsampling (and after resize if enabled)
      - flatten
    """
    name = canonical_name(name)

    # 1) Load raw arrays
    X, y = _load_source_arrays(name, seed=seed, labels_only=labels_only, out_hw=out_hw, grayscale=grayscale)
    y = np.asarray(y, dtype=int)

    if labels_only:
        return np.empty((0, 0), dtype=np.float32), y, (0, 0)

    # 2) Optional class filtering
    if classes is not None:
        classes = np.asarray(classes, dtype=int)
        mask = np.isin(y, classes)
        X, y = X[mask], y[mask]
        y = y.astype(int, copy=False)

    # 3) Normalize to [0,1]
    X = to_float01(X)

    # 4) Grayscale (always output (N,H,W))
    X = ensure_grayscale(X, grayscale=grayscale)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,H,W) after grayscale. Got shape={X.shape}")

    # 5) Optional resize
    if out_hw is not None:
        X = resize_to_hw(X, out_hw)

    # 6) Balanced subsample
    if n_samples is not None:
        X, y = stratified_subsample_strict(X, y, n_samples=int(n_samples), seed=seed)

    # 7) Apply variant
    X = apply_variant(X, variant=variant, seed=seed, noise_sigma=noise_sigma)

    # 8) Flatten
    H, W = int(X.shape[1]), int(X.shape[2])
    X_flat = X.reshape(len(X), -1).astype(np.float32, copy=False)
    return X_flat, y.astype(int, copy=False), (H, W)


# -------------------------
# Helpers
# -------------------------

def canonical_name(name: str) -> str:
    s = str(name).lower().strip()

    # existing
    s = s.replace("emnist_byclass", "emnist/byclass") \
         .replace("emnist-byclass", "emnist/byclass") \
         .replace("emnistbyclass", "emnist/byclass")
    s = s.replace("svhn-cropped", "svhn_cropped") \
         .replace("svhncropped", "svhn_cropped")

    # new aliases
    s = s.replace("stl-10", "stl10").replace("stl_10", "stl10")
    s = s.replace("quickdraw", "quickdraw_bitmap").replace("quick_draw", "quickdraw_bitmap")
    s = s.replace("german_traffic_sign", "gtsrb").replace("german-traffic-sign", "gtsrb")
    s = s.replace("kth-tips2", "kth_tips2").replace("kth_tips-2", "kth_tips2").replace("kth_tips_2", "kth_tips2")
    s = s.replace("curett", "curet").replace("cu-ret", "curet")
    if s in {"eurosat", "euro_sat"}:
        s = "eurosat"
    if s in {"binary_alphadigits", "binary-alphadigits", "alphadigits"}:
        s = "binary_alphadigits"
    if s in {"geometric_shapes", "geometric-shapes", "shapes"}:
        s = "geometric_shapes"
        # --- new aliases (TFDS image classification) ---
    if s in {"imagenette", "imagenette/160px", "imagenette160", "image_nette"}:
        s = "imagenette"
    if s in {"oxford_iiit_pet", "oxford-iiit-pet", "oxfordiiitpet", "pets", "oxford_pets"}:
        s = "oxford_iiit_pet"
    if s in {"tf_flowers", "tfflowers", "tensorflow_flowers", "flowers"}:
        s = "tf_flowers"
    if s in {"caltech101", "caltech-101", "caltech_101"}:
        s = "caltech101"
    if s in {"cats_vs_dogs", "cats-vs-dogs", "catsvsdogs", "gatos_vs_perros", "gatos-vs-perros", "gatosperros"}:
        s = "cats_vs_dogs"

    return s


def to_float01(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(np.float32) / 255.0
    else:
        X = X.astype(np.float32)
        if np.nanmax(X) > 1.5:
            X = X / 255.0
    return np.clip(X, 0.0, 1.0).astype(np.float32, copy=False)


def ensure_grayscale(X: np.ndarray, grayscale: bool = True) -> np.ndarray:
    """
    Ensure output shape is (N,H,W). If X is RGB, convert with luminance.
    Always returns grayscale for consistent downstream processing.
    """
    X = np.asarray(X)

    if X.ndim == 3:
        return X.astype(np.float32, copy=False)

    if X.ndim != 4:
        raise ValueError(f"Expected X in (N,H,W) or (N,H,W,C). Got shape={X.shape}")

    C = X.shape[-1]
    if C == 1:
        return X[..., 0].astype(np.float32, copy=False)

    if C == 3:
        return (0.2989 * X[..., 0] + 0.5870 * X[..., 1] + 0.1140 * X[..., 2]).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported channel count C={C} for grayscale conversion. shape={X.shape}")


def resize_to_hw(X: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize (N,H,W) -> (N,H_out,W_out) using TF bilinear + antialias.
    Only called if out_hw is not None.
    """
    H_out, W_out = map(int, out_hw)
    if X.ndim != 3:
        raise ValueError(f"resize_to_hw expects (N,H,W). Got {X.shape}")

    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError("tensorflow is required for resizing. Install tensorflow.") from e

    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    X_tf = tf.expand_dims(X_tf, axis=-1)  # (N,H,W,1)
    X_tf = tf.image.resize(X_tf, [H_out, W_out], method="bilinear", antialias=True)
    X_tf = tf.squeeze(X_tf, axis=-1)      # (N,H,W)
    return X_tf.numpy().astype(np.float32, copy=False)

def jitter_shift(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    max_shift: int,
    mode: str = "reflect",   # "reflect" (recomendado) o "wrap"
) -> np.ndarray:
    """
    Apply per-sample random translation jitter to images.

    X: (N,H,W) float32 in [0,1]
    max_shift: maximum absolute shift in pixels (>=0)
    mode:
      - "wrap": circular shift (np.roll)  (fast, but introduces wrap artifacts)
      - "reflect": pad-reflect then crop (more realistic, no wrap)
    """
    X = np.asarray(X, dtype=np.float32)
    if max_shift <= 0:
        return X

    N, H, W = X.shape
    k = int(max_shift)

    if mode not in {"wrap", "reflect"}:
        raise ValueError("jitter mode must be 'wrap' or 'reflect'")

    # sample shifts independently per image
    dx = rng.integers(-k, k + 1, size=N)  # vertical shift (rows)
    dy = rng.integers(-k, k + 1, size=N)  # horizontal shift (cols)

    if mode == "wrap":
        Xo = X.copy()
        for i in range(N):
            if dx[i] != 0:
                Xo[i] = np.roll(Xo[i], shift=int(dx[i]), axis=0)
            if dy[i] != 0:
                Xo[i] = np.roll(Xo[i], shift=int(dy[i]), axis=1)
        return Xo

    # reflect mode: pad then crop
    pad = k
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    Xo = np.empty_like(X)

    for i in range(N):
        # original top-left in padded coordinates is (pad, pad)
        r0 = pad + int(dx[i])
        c0 = pad + int(dy[i])
        Xo[i] = Xp[i, r0:r0 + H, c0:c0 + W]

    return Xo


def apply_variant(
    X: np.ndarray,
    *,
    variant: str,
    seed: int,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """
    Variant perturbations on (N,H,W) in [0,1].

    - raw: identity
    - gauss/noise: additive Gaussian noise with std=noise_sigma + clip to [0,1]
    - jitter/shift: random translation up to max_shift pixels (noise_sigma interpreted as pixels)
        * mode is fixed here to 'reflect' (recommended). Change to 'wrap' if you prefer.
    """
    variant = str(variant).lower().strip()

    if variant == "raw":
        return X.astype(np.float32, copy=False)

    # Gaussian additive noise (pixel i.i.d.)
    if variant in {"gauss", "noise"}:
        sigma = float(noise_sigma)
        if sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        if sigma == 0.0:
            return X.astype(np.float32, copy=False)
        rng = np.random.default_rng(seed)
        eps = rng.normal(loc=0.0, scale=sigma, size=X.shape).astype(np.float32, copy=False)
        Xn = X.astype(np.float32, copy=False) + eps
        return np.clip(Xn, 0.0, 1.0).astype(np.float32, copy=False)

    # Jitter / shift (geometric noise)
    if variant in {"jitter", "shift"}:
        max_shift = int(round(float(noise_sigma)))
        if max_shift < 0:
            raise ValueError("noise_sigma for jitter must be >= 0 (interpreted as max_shift pixels)")
        if max_shift == 0:
            return X.astype(np.float32, copy=False)
        rng = np.random.default_rng(seed)
        Xj = jitter_shift(X.astype(np.float32, copy=False), rng=rng, max_shift=max_shift, mode="reflect")
        # Already in [0,1]
        return Xj.astype(np.float32, copy=False)

    raise ValueError(f"Unknown variant='{variant}'. Use 'raw', 'gauss', or 'jitter'.")


def stratified_subsample_strict(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int,
    seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strict-ish stratified sampling:
    - target N, distributed as evenly as possible across present classes
    - if a class has insufficient samples, we redistribute shortfall across remaining classes
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)

    classes = np.unique(y)
    C = len(classes)
    if C == 0:
        raise ValueError("No samples available.")

    n_target = min(int(n_samples), len(y))
    base = n_target // C
    rem = n_target % C

    classes_sorted = np.sort(classes)
    quota = {int(c): base for c in classes_sorted}
    for c in classes_sorted[:rem]:
        quota[int(c)] += 1

    idx_by_c: Dict[int, np.ndarray] = {}
    for c in classes_sorted:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        idx_by_c[int(c)] = idx

    chosen: List[int] = []
    shortfall = 0

    for c in classes_sorted:
        idx = idx_by_c[int(c)]
        take = min(quota[int(c)], len(idx))
        chosen.extend(idx[:take].tolist())
        if take < quota[int(c)]:
            shortfall += (quota[int(c)] - take)
        idx_by_c[int(c)] = idx[take:]

    if shortfall > 0:
        remaining = np.concatenate(
            [idx_by_c[int(c)] for c in classes_sorted if len(idx_by_c[int(c)]) > 0],
            axis=0
        ) if any(len(idx_by_c[int(c)]) > 0 for c in classes_sorted) else np.array([], dtype=int)

        if len(remaining) > 0:
            rng.shuffle(remaining)
            extra_take = min(shortfall, len(remaining))
            chosen.extend(remaining[:extra_take].tolist())

    chosen = np.array(chosen, dtype=int)
    rng.shuffle(chosen)
    chosen = chosen[:n_target]
    return X[chosen], y[chosen]


# -------------------------
# Class selection rule (kept)
# -------------------------

def choose_num_classes(
    *,
    n_samples: int,
    samples_per_class: int = 10,
    class_cap: int = 20,
    n_classes_available: int,
) -> int:
    """
    C = min(C_cap, floor(N/s), C_available)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be > 0")
    return int(min(int(class_cap), int(n_samples // samples_per_class), int(n_classes_available)))


def select_classes_deterministic(y_all: np.ndarray, C: int, seed: int) -> list[int]:
    """
    Deterministic class subset selection for reproducibility.
    """
    classes_av = np.unique(np.asarray(y_all, dtype=int))
    C = min(int(C), len(classes_av))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(classes_av, size=int(C), replace=False)
    return [int(x) for x in chosen]


def _encode_labels_to_int(y):
    """
    Convert labels (possibly strings) to deterministic integer codes 0..C-1.
    Keeps ordering stable by sorting unique labels as strings.
    """
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(int, copy=False)

    # Convert to string labels
    ys = y.astype(str)
    classes = np.unique(ys)
    classes_sorted = np.sort(classes)
    mapping = {c: i for i, c in enumerate(classes_sorted)}
    return np.array([mapping[v] for v in ys], dtype=int)


# -------------------------
# Source loaders
# -------------------------

def _load_source_arrays(name: str, *, seed: int, labels_only: bool, out_hw:tuple[int, int], grayscale:bool):
    if name in ("olivetti", "digits"):
        return _load_sklearn(name, labels_only=labels_only)

    if name in ("mnist", "fashion_mnist", "cifar10"):
        return _load_keras(name, labels_only=labels_only)

    if name in ("emnist/byclass", "svhn_cropped"):
        return _load_tfds(name, seed=seed, labels_only=labels_only)

    if name in ("usps", "optdigits", "kmnist"):
        return _load_openml(name, seed=seed, labels_only=labels_only)

    if name in ("pathmnist", "dermamnist", "organcmnist", "octmnist", "retinamnist", "breastmnist", "pneumoniamnist"):
        return _load_medmnist(name, seed=seed, labels_only=labels_only)
    
    # -------------------
    # NEW DATASETS
    # -------------------
    if name in ("dtd", "omniglot", "stl10"):
        return _load_tfds_streaming(name, seed=seed, labels_only=labels_only, out_hw=out_hw, to_grayscale=False)


    if name == "gtsrb":
        return _load_torchvision_gtsrb(
            labels_only=labels_only,
            out_hw=out_hw,          # clave
        )

    if name in ("kth_tips2"):
        return _load_kaggle_imagefolder(name,
            seed=seed,
            labels_only=labels_only,
            out_hw=out_hw,          # <-- clave
            to_grayscale=False      # (déjalo False, ya conviertes luego)
        )
    if name == "fer2013":
        return _load_fer2013_kaggle(labels_only=labels_only)
    if name == "eurosat":
        return _load_eurosat_kaggle(labels_only=labels_only)

    if name == "binary_alpha_digits":
        return _load_tfds_streaming(
            name,
            seed=seed,
            labels_only=labels_only,
            out_hw=out_hw,              # opcional; tu pipeline lo hará después igual
            to_grayscale=False
        )
    if name in ("geometric_shapes",):
        return _load_geometric_shapes(seed=seed, labels_only=labels_only)
    if name in ("imagenette", "oxford_iiit_pet", "tf_flowers", "caltech101", "cats_vs_dogs"):
        return _load_tfds_streaming(
            name,
            seed=seed,
            labels_only=labels_only,
            out_hw=out_hw,          # if None, loader will choose a safe default
            to_grayscale=False      # keep RGB here; pipeline converts later
        )

    raise ValueError(f"Unknown dataset name: {name}")




def _load_sklearn(name: str, *, labels_only: bool) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_olivetti_faces, load_digits

    if name == "olivetti":
        data = fetch_olivetti_faces()
        X = data.images
        y = data.target
    else:
        data = load_digits()
        X = data.images
        y = data.target

    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), np.asarray(y, dtype=int)

    return np.asarray(X), np.asarray(y, dtype=int)


def _load_keras(name: str, *, labels_only: bool) -> tuple[np.ndarray, np.ndarray]:
    from keras import datasets as kd

    if name == "mnist":
        (Xtr, ytr), (Xte, yte) = kd.mnist.load_data()
    elif name == "fashion_mnist":
        (Xtr, ytr), (Xte, yte) = kd.fashion_mnist.load_data()
    elif name == "kmnist":
        if not hasattr(kd, "kmnist"):
            raise ImportError("tensorflow.keras.datasets.kmnist not available in this TF install.")
        (Xtr, ytr), (Xte, yte) = kd.kmnist.load_data()
    elif name == "cifar10":
        (Xtr, ytr), (Xte, yte) = kd.cifar10.load_data()
        ytr = ytr.reshape(-1)
        yte = yte.reshape(-1)
    else:
        raise ValueError(name)

    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0).astype(int)

    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), y

    return X, y


def _load_tfds(name: str, *, seed: int, labels_only: bool) -> tuple[np.ndarray, np.ndarray]:
    try:
        import tensorflow_datasets as tfds
    except Exception as e:
        raise ImportError("tensorflow_datasets required for EMNIST/SVHN. Install tensorflow_datasets.") from e

    tfds_name = "emnist/byclass" if name.startswith("emnist") else "svhn_cropped"

    ds_tr = tfds.load(tfds_name, split="train", as_supervised=True)
    ds_te = tfds.load(tfds_name, split="test", as_supervised=True)
    ds = ds_tr.concatenate(ds_te)
    ds = ds.shuffle(buffer_size=4096, seed=seed, reshuffle_each_iteration=False)

    if labels_only:
        ys = []
        for _, y in tfds.as_numpy(ds):
            ys.append(int(y))
        return np.empty((0, 1, 1), dtype=np.float32), np.asarray(ys, dtype=int)

    Xs, ys = [], []
    for x, y in tfds.as_numpy(ds):
        Xs.append(x)
        ys.append(int(y))
    X = np.stack(Xs, axis=0)
    y = np.asarray(ys, dtype=int)
    return X, y

def _load_openml(name: str, *, seed: int, labels_only: bool):
    """
    OpenML datasets, cached locally by sklearn after first download.
    Returned as (N,H,W) arrays when possible.
    """
    from sklearn.datasets import fetch_openml

    rng = np.random.default_rng(seed)

    if name == "usps":
        ds = fetch_openml("usps", version=2, as_frame=False)
        X = ds.data.reshape(-1, 16, 16)
        y = ds.target.astype(int)

    elif name == "optdigits":
        ds = fetch_openml("optdigits", version=1, as_frame=False)
        X = ds.data.reshape(-1, 8, 8)
        y = ds.target.astype(int)

    elif name == "kmnist":
        # Kuzushiji-MNIST (28x28), cached by sklearn after first download
        ds = fetch_openml("Kuzushiji-MNIST", version=1, as_frame=False)
        X = ds.data.reshape(-1, 28, 28).astype(np.float32)
        y = ds.target.astype(int)

    else:
        raise ValueError(name)

    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), y

    return X.astype(np.float32), y

def _load_medmnist(name: str, *, seed: int, labels_only: bool):
    """
    MedMNIST datasets: 2D medical images
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as e:
        raise ImportError(
            "medmnist required. Install with: pip install medmnist"
        ) from e

    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])

    dataset = DataClass(split="train", download=True)
    X = dataset.imgs
    y = dataset.labels.squeeze()

    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), y.astype(int)

    # Ensure shape (N,H,W)
    if X.ndim == 4 and X.shape[-1] == 3:
        # RGB → grayscale handled later
        pass
    elif X.ndim == 4 and X.shape[-1] == 1:
        X = X[..., 0]

    return X.astype(np.float32), y.astype(int)


def _load_tfds_streaming(
    name: str,
    *,
    seed: int,
    labels_only: bool,
    out_hw: tuple[int, int] | None = None,
    to_grayscale: bool = False,
):
    """
    TFDS loader that can optionally resize images BEFORE stacking, required for
    variable-size datasets like DTD.
    """
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except Exception as e:
        raise ImportError("tensorflow + tensorflow_datasets required for TFDS streaming.") from e

    tfds_name = {
        "dtd": "dtd",
        "omniglot": "omniglot",
        "stl10": "stl10",
        "eurosat": "eurosat",
        "binary_alpha_digits": "binary_alpha_digits",

        # NEW
        "imagenette": "imagenette",
        "oxford_iiit_pet": "oxford_iiit_pet",
        "tf_flowers": "tf_flowers",
        "caltech101": "caltech101",
        "cats_vs_dogs": "cats_vs_dogs",
    }[name]

    cap = {
        "dtd": 10000,
        "stl10": 15000,
        "omniglot": 15000,
        "eurosat": 12000,
        "binary_alpha_digits": 5000,

        # NEW (caps conservadores; tú luego haces subsample estratificado)
        "imagenette": 15000,        # ~13k train + val
        "oxford_iiit_pet": 8000,     # ~7k total (train+test)
        "tf_flowers": 4000,          # 3670
        "caltech101": 12000,         # ~9k
        "cats_vs_dogs": 25000,       # 25k (ojo pesa, pero cap limita)
    }[name]


    # Load splits
    ds_tr = tfds.load(tfds_name, split="train")
    ds_parts = [ds_tr]
    for sp in ("test", "validation"):
        try:
            ds_parts.append(tfds.load(tfds_name, split=sp))
        except Exception:
            pass
    ds = ds_parts[0]
    for d in ds_parts[1:]:
        ds = ds.concatenate(d)

    ds = ds.shuffle(buffer_size=4096, seed=seed, reshuffle_each_iteration=False)

    # Extractor for dict examples
    def extract_xy(ex):
        if isinstance(ex, dict):
            x = ex.get("image", None)
            if x is None:
                raise KeyError(f"{tfds_name}: missing 'image' in keys={list(ex.keys())}")
            if "label" in ex:
                y = ex["label"]
            elif "labels" in ex:
                y = ex["labels"]
            else:
                raise KeyError(f"{tfds_name}: missing label key in keys={list(ex.keys())}")
            return x, y
        # tuple supervised
        return ex

    # Try supervised first, fallback to dict
    supervised = False
    try:
        ds_sup_tr = tfds.load(tfds_name, split="train", as_supervised=True)
        ds_sup_parts = [ds_sup_tr]
        for sp in ("test", "validation"):
            try:
                ds_sup_parts.append(tfds.load(tfds_name, split=sp, as_supervised=True))
            except Exception:
                pass
        ds_sup = ds_sup_parts[0]
        for d in ds_sup_parts[1:]:
            ds_sup = ds_sup.concatenate(d)
        ds_use = ds_sup.shuffle(buffer_size=4096, seed=seed, reshuffle_each_iteration=False)
        it = tfds.as_numpy(ds_use.take(cap))
        supervised = True
    except Exception:
        it = tfds.as_numpy(ds.take(cap))
        supervised = False

    # Optional resize helper (expects numpy arrays)
    def maybe_resize_np(x_np: np.ndarray) -> np.ndarray:
        """
        Force all TFDS images to a fixed shape before stacking.
        Output will be:
        - (H_out, W_out) if to_grayscale=True
        - (H_out, W_out, 3) if to_grayscale=False
        """
        x_np = np.asarray(x_np)

        # Normalize input to (H,W,C)
        if x_np.ndim == 2:
            x_np = x_np[..., None]  # (H,W,1)
        elif x_np.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected image ndim={x_np.ndim} shape={x_np.shape}")

        # Handle channels: force to 3-channel RGB-ish
        C = x_np.shape[-1]
        if C == 1:
            x_np = np.repeat(x_np, 3, axis=-1)  # (H,W,3)
        elif C == 3:
            pass
        elif C == 4:
            x_np = x_np[..., :3]  # drop alpha
        else:
            # weird channel count -> take first 3 (or replicate if <3)
            if C > 3:
                x_np = x_np[..., :3]
            else:
                x_np = np.repeat(x_np[..., :1], 3, axis=-1)

        # If no resize requested, still ensure consistent shape across samples:
        # DTD is variable-size, so we MUST resize for it (and any dataset with variable size).
        # If caller didn't request out_hw, choose a safe default to allow stacking.
        # (Pipeline may resize again later; this is just to make shapes consistent.)
        if out_hw is None:
            if name == "imagenette":
                H_out, W_out = 160, 160
            else:
                H_out, W_out = 128, 128
        else:
            H_out, W_out = map(int, out_hw)


        H_out, W_out = map(int, out_hw)

        import tensorflow as tf
        x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
        x_tf = tf.image.resize(x_tf, [H_out, W_out], method="bilinear", antialias=True)
        x_rs = x_tf.numpy()  # (H_out,W_out,3) float32

        if to_grayscale:
            # luminance -> (H_out,W_out)
            x_rs = 0.2989 * x_rs[..., 0] + 0.5870 * x_rs[..., 1] + 0.1140 * x_rs[..., 2]
            return x_rs.astype(np.float32, copy=False)

        return x_rs.astype(np.float32, copy=False)


    # Materialize
    if labels_only:
        ys = []
        for ex in it:
            x, y = ex if supervised else extract_xy(ex)
            ys.append(int(np.asarray(y)))
        return np.empty((0, 1, 1), dtype=np.float32), np.asarray(ys, dtype=int)

    Xs, ys = [], []
    for ex in it:
        x, y = ex if supervised else extract_xy(ex)

        # x is uint8 typically; keep it as numpy, then resize if needed
        x = np.asarray(x)
        x = maybe_resize_np(x)

        Xs.append(x)
        ys.append(int(np.asarray(y)))

    # Now stacking is safe because we enforced fixed out_hw (or dataset is fixed-size)
    X = np.stack(Xs, axis=0)
    y = np.asarray(ys, dtype=int)
    return X, y

def _load_torchvision_gtsrb(*, labels_only: bool, out_hw: tuple[int, int] | None = None):
    try:
        from torchvision.datasets import GTSRB
    except Exception as e:
        raise ImportError("torchvision required for GTSRB. Install torchvision.") from e

    import os
    import numpy as np
    from PIL import Image

    root = os.path.join(os.path.expanduser("~"), ".cache", "torchvision")

    ds_tr = GTSRB(root=root, split="train", download=True)
    ds_te = GTSRB(root=root, split="test", download=True)

    # cap to keep RAM bounded (you subsample later anyway)
    cap = 20000

    if out_hw is not None:
        H_out, W_out = map(int, out_hw)

    def materialize(ds, cap=None):
        Xs, ys = [], []
        n = len(ds) if cap is None else min(len(ds), int(cap))
        for i in range(n):
            img, y = ds[i]  # PIL image
            img = img.convert("RGB")
            if out_hw is not None:
                img = img.resize((W_out, H_out), resample=Image.Resampling.LANCZOS)
            Xs.append(np.array(img, dtype=np.uint8))  # (H_out,W_out,3)
            ys.append(int(y))
        return np.stack(Xs, axis=0), np.asarray(ys, dtype=int)

    if labels_only:
        _, ytr = materialize(ds_tr, cap=cap)
        _, yte = materialize(ds_te, cap=cap)
        y = np.concatenate([ytr, yte], axis=0)
        return np.empty((0, 1, 1), dtype=np.float32), y

    Xtr, ytr = materialize(ds_tr, cap=cap)
    Xte, yte = materialize(ds_te, cap=cap)
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    return X, y

def _load_kaggle_imagefolder(
    name: str,
    *,
    seed: int,
    labels_only: bool,
    out_hw: tuple[int, int] | None = None,
    to_grayscale: bool = False,
):
    """
    KaggleHub-based loader for datasets delivered as folders of images.

    Expected: root/<class_name>/*.{png|jpg|jpeg|bmp}

    If out_hw is given, images are resized BEFORE stacking (required when original
    images have variable sizes, e.g., KTH-TIPS2).
    """
    import kagglehub
    from pathlib import Path
    from PIL import Image
    import numpy as np

    kaggle_id = {
        "kth_tips2": "ag3ntsp1d3rx/kth-tips-2",
        "curet": "smohsensadeghi/curet-dataset",
    }[name]

    base_path = kagglehub.dataset_download(kaggle_id)
    base = Path(base_path)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    # Find an ImageFolder-like root: a dir whose subdirs contain images
    root = None
    for p in sorted([d for d in base.rglob("*") if d.is_dir()], key=lambda x: len(str(x))):
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        ok = False
        for sd in subdirs:
            if any(f.suffix.lower() in exts for f in sd.iterdir() if f.is_file()):
                ok = True
                break
        if ok:
            root = p
            break

    if root is None:
        raise FileNotFoundError(f"Could not find ImageFolder-like structure under {base_path}")

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Collect file list
    files, ys = [], []
    for c in classes:
        for f in (root / c).rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                files.append(f)
                ys.append(class_to_idx[c])

    ys = np.asarray(ys, dtype=int)
    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), ys

    # Cap to keep RAM sane (you later stratified_subsample anyway)
    cap = 20000
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    idx = idx[:min(len(idx), cap)]

    # Resize params
    if out_hw is not None:
        H_out, W_out = map(int, out_hw)

    Xs = []
    ysel = ys[idx]

    for i in idx:
        img = Image.open(files[i])

        # Always go through RGB so downstream grayscale conversion works consistently
        img = img.convert("RGB")

        if out_hw is not None:
            # PIL resize (antialias)
            img = img.resize((W_out, H_out), resample=Image.Resampling.LANCZOS)

        x = np.asarray(img, dtype=np.uint8)  # (H,W,3) now fixed if resized

        if to_grayscale:
            # luminance -> (H,W)
            x = (0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]).astype(np.uint8)

        Xs.append(x)

    X = np.stack(Xs, axis=0)
    return X, ysel

def _load_binary_alphadigits_openml(*, seed: int, labels_only: bool):
    import numpy as np
    from sklearn.datasets import fetch_openml

    # No fixed version: OpenML changes.
    ds = fetch_openml("BinaryAlphaDigits", as_frame=False)
    X = ds.data.astype(np.float32)
    y = _encode_labels_to_int(ds.target)

    # Original shape is (20,16)
    X = X.reshape(-1, 20, 16)

    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), y

    return X, y

def _load_geometric_shapes(*, seed: int, labels_only: bool):
    """
    Synthetic geometric shapes: circle, square, triangle, cross, X.
    Returns uint8 images in [0,255] with shape (N,32,32).
    """
    import numpy as np
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)

    classes = ["circle", "square", "triangle", "cross", "x"]
    C = len(classes)
    per_class = 2000  # materialize; you'll subsample later anyway
    N = C * per_class

    y = np.repeat(np.arange(C), per_class).astype(int)

    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), y

    X = np.empty((N, 32, 32), dtype=np.uint8)

    idx = 0
    for c in range(C):
        for _ in range(per_class):
            img = Image.new("L", (32, 32), color=0)
            draw = ImageDraw.Draw(img)

            # random size/position
            r = int(rng.integers(6, 13))
            cx = int(rng.integers(10, 22))
            cy = int(rng.integers(10, 22))

            if c == 0:  # circle
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=255, fill=255)
            elif c == 1:  # square
                draw.rectangle((cx - r, cy - r, cx + r, cy + r), outline=255, fill=255)
            elif c == 2:  # triangle
                pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
                draw.polygon(pts, outline=255, fill=255)
            elif c == 3:  # cross
                t = int(rng.integers(2, 4))
                draw.rectangle((cx - t, cy - r, cx + t, cy + r), fill=255)
                draw.rectangle((cx - r, cy - t, cx + r, cy + t), fill=255)
            else:  # X
                t = int(rng.integers(1, 3))
                draw.line((cx - r, cy - r, cx + r, cy + r), fill=255, width=t)
                draw.line((cx - r, cy + r, cx + r, cy - r), fill=255, width=t)

            X[idx] = np.array(img, dtype=np.uint8)
            idx += 1

    # shuffle deterministically
    p = rng.permutation(N)
    return X[p], y[p]

def _load_fer2013_kaggle(*, labels_only: bool):
    """
    FER-2013 via KaggleHub for dataset `msambare/fer2013`, which is ImageFolder-like:
      root/train/<class>/*.jpg
      root/test/<class>/*.jpg

    Returns:
      X: (N,H,W) float32 (grayscale) or (empty) if labels_only
      y: (N,) int labels (0..C-1, deterministic by sorted class names)
    """
    import kagglehub
    import numpy as np
    from pathlib import Path
    from PIL import Image

    base_path = kagglehub.dataset_download("msambare/fer2013")
    base = Path(base_path)

    train_dir = base / "train"
    test_dir = base / "test"

    if not train_dir.exists() or not test_dir.exists():
        # helpful debug
        top = sorted([p.name for p in base.iterdir()]) if base.exists() else []
        raise FileNotFoundError(
            f"Expected 'train' and 'test' folders under {base_path}. Top-level entries: {top}"
        )

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # classes from train folder (deterministic)
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subfolders found under {train_dir}")

    classes = sorted([d.name for d in class_dirs])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # gather files from train + test
    files = []
    ys = []

    def add_split(split_dir: Path):
        for c in classes:
            d = split_dir / c
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    files.append(f)
                    ys.append(class_to_idx[c])

    add_split(train_dir)
    add_split(test_dir)

    y = np.asarray(ys, dtype=int)

    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), y

    # Materialize images
    # FER images should already be 48x48; still, we enforce grayscale consistently.
    Xs = []
    for f in files:
        img = Image.open(f).convert("L")  # grayscale
        Xs.append(np.asarray(img, dtype=np.uint8))  # (H,W)

    # If any image is not same size, stack will fail; handle by resizing to 48x48 as fallback.
    try:
        X = np.stack(Xs, axis=0)
    except ValueError:
        # fallback: force to 48x48 then stack
        Xs2 = []
        for arr, f in zip(Xs, files):
            if arr.ndim != 2:
                arr = arr.squeeze()
            img = Image.fromarray(arr)
            img = img.resize((48, 48), resample=Image.Resampling.BILINEAR)
            Xs2.append(np.asarray(img, dtype=np.uint8))
        X = np.stack(Xs2, axis=0)

    return X.astype(np.float32), y

def _load_eurosat_kaggle(*, labels_only: bool):
    """
    EuroSAT via KaggleHub, loaded as ImageFolder.

    Returns:
      X: (N,H,W,3) uint8 (or empty if labels_only)
      y: (N,) int labels (deterministic by sorted folder names)
    """
    import kagglehub
    import numpy as np
    from pathlib import Path
    from PIL import Image

    # Kaggle dataset id (widely used mirror of EuroSAT RGB)
    # If this specific one ever changes, the root-finding logic still works as long as it is ImageFolder-like.
    base_path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
    base = Path(base_path)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Find ImageFolder root: dir whose subdirs contain images
    root = None
    for p in sorted([d for d in base.rglob("*") if d.is_dir()], key=lambda x: len(str(x))):
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        ok = False
        for sd in subdirs:
            if any(f.is_file() and f.suffix.lower() in exts for f in sd.iterdir()):
                ok = True
                break
        if ok:
            root = p
            break

    if root is None:
        top = sorted([p.name for p in base.iterdir()]) if base.exists() else []
        raise FileNotFoundError(
            f"Could not find ImageFolder-like structure under {base_path}. "
            f"Top-level entries: {top[:50]}"
        )

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"No class folders found under {root}")

    class_to_idx = {c: i for i, c in enumerate(classes)}

    files, ys = [], []
    for c in classes:
        d = root / c
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                files.append(f)
                ys.append(class_to_idx[c])

    y = np.asarray(ys, dtype=int)
    if labels_only:
        return np.empty((0, 1, 1), dtype=np.float32), y

    # Cap (EuroSAT is not huge, but keep it bounded; you subsample later anyway)
    cap = 20000
    rng = np.random.default_rng(123)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    idx = idx[:min(len(idx), cap)]

    Xs = []
    ysel = y[idx]
    for i in idx:
        img = Image.open(files[i]).convert("RGB")
        Xs.append(np.asarray(img, dtype=np.uint8))  # (H,W,3) variable, but your pipeline resizes later

    # NOTE: do NOT np.stack if sizes vary; instead, pad-resize now OR stack after resizing.
    # Easiest: force-resize to 64x64 here (EuroSAT standard), then stack.
    Xs2 = []
    for arr in Xs:
        img = Image.fromarray(arr).resize((64, 64), resample=Image.Resampling.BILINEAR)
        Xs2.append(np.asarray(img, dtype=np.uint8))
    X = np.stack(Xs2, axis=0)  # (N,64,64,3)

    return X, ysel
