# src/quantum/pes_featuremap.py

import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate

def pad_to_pow2(z: np.ndarray, pad_value: complex = 1.0 + 0.0j):
    z = np.asarray(z, dtype=np.complex128).ravel()
    d = z.size
    if d == 0:
        raise ValueError("Vector vacío.")

    n_qubits = int(math.ceil(math.log2(d)))
    m = 1 << n_qubits

    if m == d:
        return z, n_qubits

    z_pad = np.empty(m, dtype=np.complex128)
    z_pad[:d] = z
    z_pad[d:] = pad_value  # <<< padding neutral

    return z_pad, n_qubits



def build_pes_featuremap_circuit( z: np.ndarray,r: float | None = None, ) -> QuantumCircuit:
    """
    PES feature map con qubit global opcional.
    - z: vector complejo |z_j| = 1
    - r: norma global (energía FFT)
    """

    # -----------------------------
    # 1) Parte local (PES estándar)
    # -----------------------------
    z_pad, n_local = pad_to_pow2(z)
    z_pad = z_pad / (np.abs(z_pad) + 1e-12)
    #if not use_global_qubit:
    qr = QuantumRegister(n_local, "q")
    qc = QuantumCircuit(qr)
    qc.h(qr)
    qc.append(DiagonalGate(z_pad.tolist()), qr)

    return qc
"""
    # -----------------------------
    # 2) Fase global (norma)
    # -----------------------------
    if r is None:
        Phi = 0.0
    else:
        Phi = np.arctan(r)

    global_phase = complex(np.cos(Phi), np.sin(Phi))

    # -----------------------------
    # 3) Circuito con qubit global
    # -----------------------------
    qr = QuantumRegister(n_local + 1, "q")
    qc = QuantumCircuit(qr)

    qg = qr[0]
    ql = qr[1:]

    qc.h(qr)
    qc.append(DiagonalGate(z_pad.tolist()), ql)
    qc.append(DiagonalGate([1.0 + 0.0j, global_phase]), [qg])

    return qc
"""

