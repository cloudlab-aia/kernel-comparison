# src/quantum/angle_featuremap.py

import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister


def normalize_vector(x):
    x = np.asarray(x, float)
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError("Vector no puede ser cero.")
    return x / norm


def build_angle_featuremap_circuit(x: np.ndarray) -> QuantumCircuit:
    """
    Angle Encoding Feature Map:
       |ψ(x)⟩ = CZ_layer · AngleEncoding(x)

    Donde AngleEncoding(x) aplica rotaciones RY(x[i]) por qubit.

    Requisitos:
      - len(x) = n qubits
    """

    x = normalize_vector(x)
    n = len(x)

    # Registro de n qubits
    qr = QuantumRegister(n, "data")
    qc = QuantumCircuit(qr, name="AngleFeatureMap")

    # ------------------------------------
    # 1) ANGLE ENCODING: RY por componente
    # ------------------------------------
    for i in range(n):
        qc.ry(x[i], qr[i])

    # ------------------------------------
    # 2) Feature Map (MISMO que PES y AE)
    # ------------------------------------


    return qc
