# src/quantum/pes_kernel.py

from .pes_featuremap import build_pes_featuremap_circuit
from src.quantum.common_kernel_ops import (
    build_gram_matrix_generic,
    build_cross_gram_matrix_generic,
)

def build_gram_matrix(
    X,
    r,
    shots: int = 2048,
    seed: int = 123,
    hadamard: bool = False,
    n_jobs: int = 20,
    prefer: str = "processes",
    do_transpile: bool = False,
):
    # Pedimos coste internamente, pero devolvemos SOLO K para que common_eval no se rompa
    K, cost = build_gram_matrix_generic(
        X,
        prep_fn=lambda z, ri: build_pes_featuremap_circuit(z, ri),
        aux_data=r,
        shots=shots,
        seed=seed,
        desc="PES",
        n_jobs=n_jobs,
        hadamard=hadamard,
        return_cost=True,
        prefer=prefer,
        do_transpile=do_transpile,
    )
    return K  # <- clave


def build_gram_matrix_with_cost(
    X,
    r,
    shots: int = 2048,
    seed: int = 123,
    hadamard: bool = False,
    n_jobs: int = 20,
    prefer: str = "processes",
    do_transpile: bool = False,
):
    # Si en algún sitio SÍ quieres el coste
    return build_gram_matrix_generic(
        X,
        prep_fn=lambda z, ri: build_pes_featuremap_circuit(z, ri),
        aux_data=r,
        shots=shots,
        seed=seed,
        desc="PES",
        n_jobs=n_jobs,
        hadamard=hadamard,
        return_cost=True,
        prefer=prefer,
        do_transpile=do_transpile,
    )


def build_cross_gram_matrix(
    ZA,
    normsA,
    ZB,
    normsB,
    shots: int = 2048,
    seed: int = 123,
    hadamard: bool = False,
    n_jobs: int = 20,
    prefer: str = "processes",
    do_transpile: bool = False,
):
    return build_cross_gram_matrix_generic(
        ZA,
        ZB,
        prep_fn=lambda z, ri: build_pes_featuremap_circuit(z, ri),
        aux_data_A=normsA,
        aux_data_B=normsB,
        shots=shots,
        seed=seed,
        n_jobs=n_jobs,
        hadamard=hadamard,
        desc="PES_CROSS",
        prefer=prefer,
        do_transpile=do_transpile,
    )
