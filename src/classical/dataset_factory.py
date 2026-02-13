import numpy as np
from src.classical.real_2d_datasets import (
    load_real_2d_dataset,
    choose_num_classes,
    select_classes_deterministic,
)

REAL_2D = {
    "olivetti", "cifar10", "digits",
    "emnist/byclass", "fashion_mnist",
    "mnist", "svhn_cropped", "pathmnist",
    "dermamnist","organcmnist","usps",
    "optdigits","kmnist", "dtd", "omniglot", 
    "stl10", "gtsrb","fer2013","eurosat","geometric_shapes",
    "octmnist", "retinamnist", "breastmnist","pneumoniamnist"
}

def load_dataset(
    name: str,
    *,
    n_samples: int,
    seed: int,
    entropy=0.2,
    n_classes=None,          # None -> apply rule
    variant="gauss",
    out_hw=(32,32),
    samples_per_class=10,
    class_cap=20,
    noise_sigma=0.10
):
    """
    Unified dataset loader.
    Returns:
        X_flat : (N, D)
        y      : (N,)
        shape  : tuple[int, ...]
    """

    name = str(name).lower().strip()

    # -------------------------------
    # REAL 2D DATASETS
    # -------------------------------
    if name in REAL_2D:
        # 1) Get labels to know C_available
        _, y_all, _ = load_real_2d_dataset(
            name,
            n_samples=None,
            classes=None,
            seed=seed,
            grayscale=True,
            out_hw=out_hw,
            variant=variant,
            labels_only=True,
            noise_sigma=noise_sigma
        )
        C_available = len(np.unique(y_all))

        # 2) Determine C by rule (unless user overrides n_classes)
        if n_classes is None:
            C = choose_num_classes(
                n_samples=n_samples,
                samples_per_class=samples_per_class,
                class_cap=class_cap,
                n_classes_available=C_available,
            )
        else:
            C = min(int(n_classes), C_available)

        # 3) Deterministically pick those C classes
        classes = select_classes_deterministic(y_all, C=C, seed=seed)

        # 4) Load the actual balanced subset (your loader already does strict stratified sampling)
        X, y, shape = load_real_2d_dataset(
            name,
            n_samples=n_samples,
            classes=classes,
            seed=seed,
            grayscale=True,
            out_hw=out_hw,
            variant=variant,
            noise_sigma=noise_sigma
        )
        return X, y, shape

    raise ValueError(f"Unknown dataset: {name}")