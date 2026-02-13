#src/classical/phase_embedding.py
import numpy as np

def real_to_phase_embedding_linear(
    Z_real: np.ndarray,
    lam: float = np.pi,
    eps: float = 1e-12,
    clip: float = 1.0,
):
    """
    Fair & stable mapping: real features -> unit-modulus complex via linear phase.

    Steps:
      - norms = ||z||_2 per sample
      - z_unit = z / norms
      - phi = lam * clip(z_unit, -clip, clip)
      - Z = exp(i * phi)

    Returns:
      Z_phase: (N, m) complex unit modulus
      norms:   (N,)
    """
    Z_real = np.asarray(Z_real, dtype=np.float64)
    norms = np.linalg.norm(Z_real, axis=1)
    norms = np.where(norms < eps, 1.0, norms)
    Z_unit = Z_real / norms[:, None]

    Z_unit = np.clip(Z_unit, -clip, clip)
    phi = lam * Z_unit
    Z_phase = np.cos(phi) + 1j * np.sin(phi)
    return Z_phase.astype(np.complex128), norms.astype(np.float64)
