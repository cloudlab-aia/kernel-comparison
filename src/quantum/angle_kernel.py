# src/quantum/angle_kernel.py

import numpy as np
from .angle_featuremap import build_angle_featuremap_circuit
from .common_kernel_ops import build_gram_matrix_generic


def build_gram_matrix(X, shots=2048, seed=123):
    X = np.asarray(X, float)
    return build_gram_matrix_generic(
        X,
        prep_fn=lambda z, r=None: build_angle_featuremap_circuit(z),
        shots=shots,
        seed=seed,
        desc="ANGLE",
        hadamard=False ,
        return_cost=True
    )
