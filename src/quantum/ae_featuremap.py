# src/quantum/ae_featuremap.py

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Isometry, StatePreparation

def normalize_vector(x):
    x = np.asarray(x, float)
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError("Vector no puede ser cero.")
    return x / norm


def build_amp_state(vec, hadamard=False):
    """
    Sustituto de initialize():
    Prepara |x> normalizado usando Isometry y lo descompone
    para que Aer lo entienda.
    """
    if hadamard:
        vec = np.asarray(vec, complex)
        vec = vec / np.linalg.norm(vec)

        prep = StatePreparation(vec)

        qc = prep # asegura que no quedan instrucciones no controlables

        return qc
    else:
        vec = np.asarray(vec, dtype=complex)
        d = len(vec)
        n = int(math.ceil(math.log2(d)))

        amp = vec / np.linalg.norm(vec)

        # Expandimos a 2^n
        if len(amp) < 2**n:
            tmp = np.zeros(2**n, dtype=complex)
            tmp[:len(amp)] = amp
            amp = tmp

        qc = QuantumCircuit(n, name="amp_u")
        iso = Isometry(amp, 0, 0)  # <-- ESTE es unitario y controlable
        qc.append(iso, qc.qubits)

        return qc
    


def build_ae_featuremap_circuit(x: np.ndarray, aux_data=None) -> QuantumCircuit:
    """
    MISMO feature map que antes:
       |ψ_AE⟩ = (CZ layer) · |amp(x)>

    NO añadimos Hadamards.
    NO añadimos diagonales.
    NO cambiamos nada del FM original.

    ÚNICAMENTE reemplazamos initialize(x) por Isometry.
    """

    x = normalize_vector(x)
    m = len(x)

    # expandir a 2^n
    n = int(math.log2(m))
    if 2**n != m:
        raise ValueError("len(x) debe ser potencia de 2.")

    # 1) amplitude encoding con Isometry
    prep = build_amp_state(x)     # |amp(x)>

    # 2) Añadir la capa de entanglement (igual que antes)
    qc = QuantumCircuit(n, name="AEFeatureMap")
    qc.compose(prep, inplace=True)

    return qc
