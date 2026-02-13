# src/quantum/common_kernel_ops.py

import os
import numpy as np
from tqdm import tqdm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
from joblib import Parallel, delayed

# =========================================================
# NOTAS IMPORTANTES (por qué esto evita el segfault)
# =========================================================
# 1) NO se pasa AerSimulator a procesos joblib (loky). Se crea dentro del worker.
# 2) Para Aer NO hace falta transpile. Evitamos transpile(backend=...) que llama a backend.target
#    (ruta exacta de tu segfault).
# 3) El backend se cachea por proceso para no recrearlo miles de veces.


# =========================================================
# SWAP TEST GENÉRICO
# =========================================================
def build_swap_test(prepA: QuantumCircuit, prepB: QuantumCircuit) -> QuantumCircuit:
    """Construye un SWAP test dado dos circuitos de preparación |ψ_A> y |ψ_B>."""
    n = prepA.num_qubits

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


# =========================================================
# HADAMARD TEST GENÉRICO
# =========================================================
def build_hadamard_test(prepA: QuantumCircuit, prepB: QuantumCircuit) -> QuantumCircuit:
    """
    Hadamard test para estimar Re⟨ψ_A|ψ_B⟩.

    De los counts:
        p0 = counts["0"]/shots
        Re⟨ψ_A|ψ_B⟩ = 2*p0 - 1
    """
    n = prepA.num_qubits
    if prepB.num_qubits != n:
        raise ValueError("prepA y prepB deben tener el mismo número de qubits")

    anc = QuantumRegister(1, "anc")
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, q, c)

    gateA = prepA.to_gate(label="A")
    gateB = prepB.to_gate(label="B")
    ctrlA = gateA.control(1)
    ctrlB = gateB.control(1)

    qc.h(anc[0])

    # anc=1 -> preparar |psi_B>
    qc.append(ctrlB, [anc[0]] + list(q))

    # aplicar A cuando anc=0: X - ctrlA - X
    qc.x(anc[0])
    qc.append(ctrlA, [anc[0]] + list(q))
    qc.x(anc[0])

    qc.h(anc[0])
    qc.measure(anc[0], c[0])
    return qc


# =========================================================
# MÉTRICAS / HELPERS
# =========================================================
def p0_from_counts(counts):
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    return counts.get("0", 0) / tot


def overlap_from_p0_swap(p0):
    # SWAP: p0 = (1 + |<A|B>|^2)/2  =>  |<A|B>|^2 = 2*p0 - 1
    return max(0.0, 2 * p0 - 1)


def overlap_from_p0_hadamard(p0):
    # Hadamard: 2*p0 - 1 = Re <A|B>
    return 2 * p0 - 1


def count_2q_gates(qc: QuantumCircuit) -> int:
    """Cuenta puertas de 2 qubits (después de transpile si quieres)."""
    c = 0
    for inst, qargs, _ in qc.data:
        if len(qargs) == 2 and inst.name not in ("measure", "barrier"):
            c += 1
    return c


# =========================================================
# BACKEND CACHE (por proceso)
# =========================================================
_BACKEND_CACHE = {}

def _get_aer_backend(seed: int) -> AerSimulator:
    """
    Devuelve un AerSimulator cacheado por proceso.

    IMPORTANTÍSIMO: esto se ejecuta dentro del worker, NO en el proceso padre.
    """
    b = _BACKEND_CACHE.get(seed)
    if b is None:
        b = AerSimulator(seed_simulator=seed)
        # Evita paralelismo interno de Aer
        b.set_options(
            max_parallel_threads=1,
            max_parallel_experiments=1
        )
        _BACKEND_CACHE[seed] = b
    return b


# =========================================================
# EJECUCIÓN (SWAP o HADAMARD) PARA UN PAR
# =========================================================
def run_test(
    prepA: QuantumCircuit,
    prepB: QuantumCircuit,
    backend: AerSimulator,
    shots: int = 2048,
    seed: int = 123,
    hadamard: bool = False,
    do_transpile: bool = False,
    transpile_basis_gates=None,
    transpile_coupling_map=None,
):
    """
    Ejecuta SWAP o Hadamard test.

    - Para Aer: por defecto do_transpile=False (MUCHO más estable y rápido).
    - Si quieres transpilar (p.ej. para contar gates), activa do_transpile y
      usa basis/coupling para evitar depender del backend.target (y evitar segfaults).
    """
    qc = build_hadamard_test(prepA, prepB) if hadamard else build_swap_test(prepA, prepB)

    if do_transpile:
        # OJO: evitamos transpile(qc, backend=backend) para no tocar backend.target.
        # Si NO das basis/coupling, transpile puede hacer decisiones por defecto.
        qc = transpile(
            qc,
            basis_gates=transpile_basis_gates,
            coupling_map=transpile_coupling_map,
            optimization_level=0,
            seed_transpiler=seed,
        )

    counts = backend.run(qc, shots=shots).result().get_counts()
    p0 = p0_from_counts(counts)

    if hadamard:
        return overlap_from_p0_hadamard(p0)
    return overlap_from_p0_swap(p0)


# =========================================================
# WORKERS (NO pasan backend desde fuera)
# =========================================================
def _pair_seed(base_seed: int, i: int, j: int) -> int:
    # semilla distinta por par: evita correlaciones y es determinista
    return int(base_seed + 1000003 * i + 9176 * j)


def compute_pair(
    i: int,
    j: int,
    prep_list,
    shots: int,
    base_seed: int,
    hadamard: bool = False,
    do_transpile: bool = False,
    transpile_basis_gates=None,
    transpile_coupling_map=None,
):
    seed = _pair_seed(base_seed, i, j)
    backend = _get_aer_backend(seed)

    prep_i = prep_list[i]
    prep_j = prep_list[j]

    kij = run_test(
        prep_i, prep_j,
        backend=backend,
        shots=shots,
        seed=seed,
        hadamard=hadamard,
        do_transpile=do_transpile,
        transpile_basis_gates=transpile_basis_gates,
        transpile_coupling_map=transpile_coupling_map,
    )
    return (i, j, kij)


def compute_pair_cross(
    i: int,
    j: int,
    prep_list_A,
    prep_list_B,
    shots: int,
    base_seed: int,
    hadamard: bool = False,
    do_transpile: bool = False,
    transpile_basis_gates=None,
    transpile_coupling_map=None,
):
    seed = _pair_seed(base_seed, i, j)
    backend = _get_aer_backend(seed)

    prep_i = prep_list_A[i]
    prep_j = prep_list_B[j]

    kij = run_test(
        prep_i, prep_j,
        backend=backend,
        shots=shots,
        seed=seed,
        hadamard=hadamard,
        do_transpile=do_transpile,
        transpile_basis_gates=transpile_basis_gates,
        transpile_coupling_map=transpile_coupling_map,
    )
    return (i, j, kij)


# =========================================================
# MATRIZ GRAM (COMÚN)
# =========================================================
def build_gram_matrix_generic(
    X,
    prep_fn,
    aux_data=None,
    shots: int = 2048,
    seed: int = 123,
    desc: str = "Kernel",
    n_jobs: int = 20,
    hadamard: bool = False,
    return_cost: bool = False,
    prefer: str = "processes",    # "processes" (loky) robusto; "threads" experimental
    do_transpile: bool = False,   # por defecto False para Aer
    transpile_basis_gates=None,
    transpile_coupling_map=None,
):
    """
    Construye matriz de kernel K (N,N) usando SWAP o Hadamard test.

    - prefer="processes": robusto (recomendado)
    - prefer="threads": evita pickling de prep_list (a veces más rápido), pero depende de thread-safety.

    Para Aer, do_transpile=False es lo más estable y rápido.
    """
    X = np.asarray(X)
    N = len(X)
    K = np.zeros((N, N), float)

    # =========================================================
    # PRECOMPILAR FEATURE MAPS
    # =========================================================
    prep_list = []
    print("\n>> Precompilando feature maps...")
    for i in tqdm(range(N), desc=f"{desc}:prep"):
        ri = aux_data[i] if aux_data is not None else None
        prep_list.append(prep_fn(X[i], ri))

    # =========================================================
    # COSTE CUÁNTICO (opcional)
    # =========================================================
    cost = None
    if return_cost and N >= 2:
        # Si quieres coste (depth / 2q), normalmente sí conviene transpilar.
        # Aquí lo dejamos como placeholder para tu función.
        # cost = circuit_cost_for_pair(prep_list[0], prep_list[1], ...)
        cost = None

    # =========================================================
    # TAREAS (i < j)
    # =========================================================
    tasks = []
    for i in range(N):
        K[i, i] = 1.0
        for j in range(i + 1, N):
            tasks.append((i, j))

    # =========================================================
    # EJECUCIÓN EN PARALELO
    # =========================================================
    print(f"\n>> Ejecutando kernel en paralelo (n_jobs={n_jobs}, prefer={prefer})...")
    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(compute_pair)(
            i, j, prep_list,
            shots=shots,
            base_seed=int(seed),
            hadamard=hadamard,
            do_transpile=do_transpile,
            transpile_basis_gates=transpile_basis_gates,
            transpile_coupling_map=transpile_coupling_map,
        )
        for (i, j) in tqdm(tasks, desc=desc)
    )

    for i, j, kij in results:
        K[i, j] = K[j, i] = kij

    if return_cost:
        return K, cost
    return K


# =========================================================
# MATRIZ GRAM CROSS (A vs B)
# =========================================================
def build_cross_gram_matrix_generic(
    XA,
    XB,
    prep_fn,
    aux_data_A=None,
    aux_data_B=None,
    shots: int = 2048,
    seed: int = 123,
    desc: str = "CrossKernel",
    n_jobs: int = 20,
    hadamard: bool = False,
    prefer: str = "processes",
    do_transpile: bool = False,
    transpile_basis_gates=None,
    transpile_coupling_map=None,
):
    """
    Build cross kernel matrix K_AB where:
      K_AB[i, j] = k(XA[i], XB[j])

    Para Aer: do_transpile=False recomendado.
    """
    XA = np.asarray(XA)
    XB = np.asarray(XB)
    NA = len(XA)
    NB = len(XB)

    K = np.zeros((NA, NB), float)

    # --- Precompile feature maps for A
    prep_list_A = []
    print("\n>> Precompilando feature maps (A)...")
    for i in tqdm(range(NA), desc=f"{desc}:prepA"):
        ri = aux_data_A[i] if aux_data_A is not None else None
        prep_list_A.append(prep_fn(XA[i], ri))

    # --- Precompile feature maps for B
    prep_list_B = []
    print("\n>> Precompilando feature maps (B)...")
    for j in tqdm(range(NB), desc=f"{desc}:prepB"):
        rj = aux_data_B[j] if aux_data_B is not None else None
        prep_list_B.append(prep_fn(XB[j], rj))

    tasks = [(i, j) for i in range(NA) for j in range(NB)]

    print(f"\n>> Ejecutando CROSS kernel en paralelo (n_jobs={n_jobs}, prefer={prefer})...")
    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(compute_pair_cross)(
            i, j, prep_list_A, prep_list_B,
            shots=shots,
            base_seed=int(seed),
            hadamard=hadamard,
            do_transpile=do_transpile,
            transpile_basis_gates=transpile_basis_gates,
            transpile_coupling_map=transpile_coupling_map,
        )
        for (i, j) in tqdm(tasks, desc=desc)
    )

    for i, j, kij in results:
        K[i, j] = kij

    return K, None
