#!/usr/bin/env python3
from __future__ import annotations

import csv
import numpy as np
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.classical.real_2d_datasets import load_real_2d_dataset
from src.classical.real_2d_datasets import choose_num_classes  # ajusta import
from src.classical.dataset_factory import REAL_2D                   # ajusta import


# ---- parámetros EXACTOS a los que uses en los experimentos ----
N_SAMPLES = 200
SAMPLES_PER_CLASS = 10
CLASS_CAP = 20
SEED = 123
OUT_HW = (32, 32)
VARIANT = "gauss"
NOISE_SIGMA = 0.10


def compute_num_classes(name: str) -> int:
    # 1) solo etiquetas → nº clases disponibles
    _, y_all, _ = load_real_2d_dataset(
        name,
        n_samples=None,
        classes=None,
        seed=SEED,
        grayscale=True,
        out_hw=OUT_HW,
        variant=VARIANT,
        noise_sigma=NOISE_SIGMA,
        labels_only=True,
    )
    C_available = len(np.unique(y_all))

    # 2) aplicar la MISMA regla que el loader
    C = choose_num_classes(
        n_samples=N_SAMPLES,
        samples_per_class=SAMPLES_PER_CLASS,
        class_cap=CLASS_CAP,
        n_classes_available=C_available,
    )
    return C


def main():
    rows = []
    for ds in sorted(REAL_2D):
        try:
            C = compute_num_classes(ds)
            rows.append((ds, C))
        except Exception as e:
            print(f"[WARN] {ds}: {e}")

    # 3) escribir CSV
    with open("classes_2d.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "num_classes"])
        writer.writerows(rows)

    print("Saved → classes_2d.csv")


if __name__ == "__main__":
    main()
