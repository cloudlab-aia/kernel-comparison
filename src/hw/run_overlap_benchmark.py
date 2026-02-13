#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HW micro-benchmark (IBM Quantum):
  - PES:  DFT/FFT -> complex phases -> DiagonalGate feature map -> SWAP test
  - ANGLE: (optional DFT/FFT) -> real features -> RY feature map -> SWAP test

Goal:
  Estimate fidelity |<psi(x)|psi(y)>|^2 on real hardware for small n_qubits,
  compare against ideal (statevector) and/or analytic PES expectation.

Outputs:
  results/hw_overlap_benchmark.csv

Notes:
  - This does NOT build a full Gram matrix / QSVM. Only overlap estimation.
  - AE is omitted on HW (Initialize/Isometry explodes). Keep AE for sim/noise elsewhere.
"""

import os
import math
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import StatePreparation

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2

# If you want SamplerV2 later, keep import:
# from qiskit_ibm_runtime import SamplerV2
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.utils.secrets_loader import Secrets


# =====================================================
# Utils: data generation
# =====================================================
def normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    n = np.linalg.norm(x)
    if n == 0:
        raise ValueError("Vector cannot be zero.")
    return x / n


def make_pair_with_cosine(d: int, rho: float, rng: np.random.Generator):
    """
    Build x,y in R^d with controlled cosine similarity:
      cos(x,y) = rho
    """
    x = rng.normal(size=d)
    x = normalize_vector(x)

    u = rng.normal(size=d)
    u = u - (u @ x) * x
    u = normalize_vector(u)

    y = rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * u
    y = normalize_vector(y)
    return x, y


# =====================================================
# FFT-based PES embedding (your function, slightly adapted)
# =====================================================
def fft_general_ab_encoding(
    X: np.ndarray,
    m: int,
    lam: float = np.pi,
    return_norm: bool = True,
):
    """
    Spectral (FFT1D) a + i b encoding for arbitrary real-valued vectors.

    Returns:
      Z: complex unit-modulus embedding (N, m)
      r: spectral energy per sample (N,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be (N,d).")

    N, d = X.shape
    if m > d:
        raise ValueError(f"m={m} cannot be > d={d}")

    F = np.fft.fft(X, axis=1)
    F = F[:, :m]

    A = np.abs(F)
    theta = np.angle(F)

    r = np.linalg.norm(A, axis=1)

    w = np.log1p(A)
    w = w / (w.max() + 1e-12)

    phi = theta + lam * w
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    Z = np.cos(phi) + 1j * np.sin(phi)

    if return_norm:
        return Z, r
    return Z


# =====================================================
# Feature maps
# =====================================================
def pad_to_pow2(z: np.ndarray):
    z = np.asarray(z, dtype=np.complex128).ravel()
    d = z.size
    if d == 0:
        raise ValueError("Empty vector.")

    n_qubits = int(math.ceil(math.log2(d)))
    m = 1 << n_qubits
    if m == d:
        return z, n_qubits

    z_pad = np.empty(m, dtype=np.complex128)
    z_pad[:d] = z

    pad_len = m - d
    phases = np.empty(pad_len, dtype=np.complex128)
    phases[0::2] = 1j
    phases[1::2] = -1j
    z_pad[d:] = phases
    return z_pad, n_qubits


def build_pes_featuremap_circuit(z: np.ndarray) -> QuantumCircuit:
    """
    |psi(z)> = Diagonal(z) H^{\otimes n} |0...0>
    with |z_j| = 1 and len(z) = 2^n (or padded).
    """
    z_pad, n = pad_to_pow2(z)
    z_pad = z_pad / (np.abs(z_pad) + 1e-12)

    qr = QuantumRegister(n, "q")
    qc = QuantumCircuit(qr, name="PES")
    qc.h(qr)
    qc.append(DiagonalGate(z_pad.tolist()), qr)
    return qc

def build_ae_featuremap_circuit(x: np.ndarray) -> QuantumCircuit:
    """
    Amplitude encoding for small n (m = 2^n).
    |psi(x)> = sum_j x_j |j>, ||x|| = 1
    """
    x = np.asarray(x, float).ravel()
    x = normalize_vector(x)

    d = x.size
    n = int(math.log2(d))
    if 2**n != d:
        raise ValueError("AE requires dimension = 2^n")

    qr = QuantumRegister(n, "q")
    qc = QuantumCircuit(qr, name="AE")

    qc.append(StatePreparation(x.tolist()), qr)
    return qc

def truncate_block_mean(x: np.ndarray, m: int) -> np.ndarray:
    """
    Deterministic truncation by block averaging.
    Maps R^d -> R^m with d divisible by m.
    """
    x = np.asarray(x, float).ravel()
    d = x.size
    if d % m != 0:
        raise ValueError(f"d={d} must be divisible by m={m}")
    b = d // m
    return np.array([x[i*b:(i+1)*b].mean() for i in range(m)])

def angle_features_from_vector(x: np.ndarray, n: int, use_dft: bool, eps=1e-12):
    """
    Build n real angles from x (real vector).
    If use_dft: use magnitudes of first n FFT coefficients.
    Else: use first n components of x.

    Map to [0, pi] for RY.
    """
    x = np.asarray(x, float).ravel()
    if use_dft:
        F = np.fft.fft(x)
        v = np.abs(F[:n])
    else:
        if x.size < n:
            raise ValueError("x too short for angle features")
        v = x[:n].copy()

    # scale to [0, pi]
    v = (v - v.min()) / (v.max() - v.min() + eps)
    return np.pi * v


def build_angle_featuremap_circuit(x: np.ndarray, use_dft: bool = False) -> QuantumCircuit:
    """
    Simple angle encoding:
      apply RY(theta_i) per qubit
    """
    n = None
    # We'll infer n from length of produced features in caller
    raise RuntimeError("Call build_angle_featuremap_circuit_from_angles instead.")


def build_angle_featuremap_circuit_from_angles(theta: np.ndarray) -> QuantumCircuit:
    theta = np.asarray(theta, float).ravel()
    n = theta.size
    qr = QuantumRegister(n, "q")
    qc = QuantumCircuit(qr, name="ANGLE")
    for i in range(n):
        qc.ry(theta[i], qr[i])
    return qc


# =====================================================
# SWAP test + postprocessing
# =====================================================
def build_swap_test(prepA: QuantumCircuit, prepB: QuantumCircuit) -> QuantumCircuit:
    n = prepA.num_qubits
    if prepB.num_qubits != n:
        raise ValueError("prepA and prepB must have same #qubits.")

    anc = QuantumRegister(1, "anc")
    qa = QuantumRegister(n, "a")
    qb = QuantumRegister(n, "b")
    c = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, qa, qb, c)
    qc.compose(prepA, qa, inplace=True)
    qc.compose(prepB, qb, inplace=True)

    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc[0])
    qc.measure(anc[0], c[0])
    return qc


def p0_from_counts(counts: dict) -> float:
    shots = sum(counts.values())
    return counts.get("0", 0) / max(1, shots)


def fidelity_from_p0_swap(p0: float) -> float:
    """
    SWAP test:
      p0 = (1 + |<psi|phi>|^2)/2
      => fidelity = |<psi|phi>|^2 = 2*p0 - 1
    """
    return float(max(0.0, min(1.0, 2.0 * p0 - 1.0)))


def ideal_fidelity_from_preps(prepA: QuantumCircuit, prepB: QuantumCircuit) -> float:
    """
    Compute ideal fidelity via statevector (small n only).
    """
    svA = Statevector.from_instruction(prepA)
    svB = Statevector.from_instruction(prepB)
    ov = np.vdot(svA.data, svB.data)
    return float(np.abs(ov) ** 2)


def pes_expected_fidelity(zA: np.ndarray, zB: np.ndarray) -> float:
    """
    For PES state:
      |psi(z)> = (1/sqrt(m)) sum_j z_j |j>
    Inner product:
      <psi(zA)|psi(zB)> = (1/m) sum_j conj(zA_j) zB_j
    Fidelity = |...|^2
    """
    zA = np.asarray(zA, complex).ravel()
    zB = np.asarray(zB, complex).ravel()
    if zA.size != zB.size:
        raise ValueError("zA and zB must have same length.")
    m = zA.size
    ov = np.mean(np.conjugate(zA) * zB)
    return float(np.abs(ov) ** 2)


# =====================================================
# IBM execution
# =====================================================
def run_circuits_on_backend(backend, circuits, shots: int, seed: int, opt_level: int):
    """
    Ejecuta una lista de circuitos en hardware real usando SamplerV2.
    Devuelve:
      - tqcs: circuitos transpilados
      - results: lista de objetos SamplerResult (uno por circuito)
    """

    # 1) Transpile (igual que antes)
    tqcs = transpile(
        circuits,
        backend=backend,
        optimization_level=opt_level,
        seed_transpiler=seed,
        translation_method="translator",
        routing_method="sabre",
    )

    # 2) Ejecutar con SamplerV2
    sampler = SamplerV2(mode=backend)
    job = sampler.run(tqcs, shots=shots)
    result = job.result()

    return tqcs, result

def extract_counts_list(result_obj, n_circuits: int):
    """
    Robust-ish counts extraction for IBM backend.run results.
    """
    counts_list = []
    for i in range(n_circuits):
        counts = result_obj.get_counts(i)
        counts_list.append(counts)
    return counts_list


def transpile_metrics(tqc: QuantumCircuit):
    ops = tqc.count_ops()
    depth = tqc.depth()
    size = tqc.size()
    # heuristic 2Q count:
    twoq = 0
    for k, v in ops.items():
        if k in ("cx", "cz", "ecr", "swap", "cswap"):
            twoq += v
    return depth, size, int(twoq), ops


# =====================================================
# Main experiment
# =====================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/hw_overlap_all_n2_marrakesh.csv")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--opt_level", type=int, default=1)
    ap.add_argument("--d_raw", type=int, default=256)
    ap.add_argument("--pairs_per_rho", type=int, default=5)
    ap.add_argument("--lam", type=float, default=np.pi)
    ap.add_argument("--angle_with_dft", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---------------------------   
    # IBM backend
    # ---------------------------
    sec = Secrets()
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=sec.qiskit_api_key,
        instance=sec.qiskit_instance
    )
    backend = service.backend(args.backend or sec.hardware_backend)
    print("Usando backend:", backend.name)

    rho_list = [-0.8, -0.4, 0.0, 0.4, 0.8]
    n_list = [2]

    rows = []
    t0 = time.time()

    for n in n_list:
        m = 1 << n
        print(f"\n=== n={n} (m={m}) ===")

        circuits = []
        meta = []

        for rho in rho_list:
            for rep in range(args.pairs_per_rho):
                x, y = make_pair_with_cosine(args.d_raw, rho, rng)

                # ===== PES =====
                Z, _ = fft_general_ab_encoding(
                    np.vstack([x, y]), m=m, lam=args.lam, return_norm=True
                )
                zA, zB = Z

                prepA = build_pes_featuremap_circuit(zA)
                prepB = build_pes_featuremap_circuit(zB)
                qc = build_swap_test(prepA, prepB)

                p0_exp = 0.5 * (1 + pes_expected_fidelity(zA, zB))

                circuits.append(qc)
                meta.append({
                    "method": "PES_DFT",
                    "rho": rho,
                    "rep": rep,
                    "p0_expected": p0_exp
                })

                # ===== ANGLE =====
                x_ang = truncate_block_mean(x, m)
                y_ang = truncate_block_mean(y, m)

                thA = angle_features_from_vector(x_ang, n=n, use_dft=False)
                thB = angle_features_from_vector(y_ang, n=n, use_dft=False)

                prepA = build_angle_featuremap_circuit_from_angles(thA)
                prepB = build_angle_featuremap_circuit_from_angles(thB)
                qc = build_swap_test(prepA, prepB)

                fid_ideal = ideal_fidelity_from_preps(prepA, prepB)
                p0_exp = 0.5 * (1 + fid_ideal)

                circuits.append(qc)
                meta.append({
                    "method": "ANGLE_raw",
                    "rho": rho,
                    "rep": rep,
                    "p0_expected": p0_exp
                })
                # ===== AE =====

                # AE truncation (same x,y as PES/ANGLE)
                x_ae = truncate_block_mean(x, m)   # m = 2^n = 8
                y_ae = truncate_block_mean(y, m)

                prepA = build_ae_featuremap_circuit(x_ae)
                prepB = build_ae_featuremap_circuit(y_ae)
                qc = build_swap_test(prepA, prepB)   # ← ESTO FALTABA


                # For AE: |<x|y>|^2 = cos(x,y)^2 = rho^2
                fid_ideal = ideal_fidelity_from_preps(prepA, prepB)
                p0_exp = 0.5 * (1 + fid_ideal)

                circuits.append(qc)
                meta.append({
                    "method": "AE",
                    "rho": rho,
                    "rep": rep,
                    "p0_expected": p0_exp
                })


        print(f"Total circuits: {len(circuits)}")

        tqcs, res = run_circuits_on_backend(
            backend=backend,
            circuits=circuits,
            shots=args.shots,
            seed=args.seed,
            opt_level=args.opt_level
        )

        exec_id = 0
        for tqc, pub_res, md in zip(tqcs, res, meta):
            exec_id += 1
            counts = pub_res.join_data().get_counts()
            p0_hw = p0_from_counts(counts)
            err = abs(p0_hw - md["p0_expected"])

            depth, size, twoq, _ = transpile_metrics(tqc)

            print(
                f"[{exec_id:03d}] {md['method']:<10} "
                f"| rho={md['rho']:+.2f} "
                f"| p0_hw={p0_hw:.3f} "
                f"| p0_exp={md['p0_expected']:.3f} "
                f"| |Δp0|={err:.3f} "
                f"| depth={depth} | 2Q≈{twoq}"
            )

            rows.append({
                "backend": backend.name,
                "method": md["method"],
                "rho": md["rho"],
                "rep": md["rep"],
                "p0_hw": p0_hw,
                "p0_expected": md["p0_expected"],
                "abs_err_p0": err,
                "depth": depth,
                "twoq": twoq,
                "shots": args.shots,
            })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nSaved: {args.out}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
