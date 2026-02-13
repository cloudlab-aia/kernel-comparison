#src/classical/baselines_matched
import numpy as np
from sklearn.decomposition import PCA
from .phase_embedding import real_to_phase_embedding_linear

def pca_fit_transform_split(X_tr, X_te, m, seed=123, lam=np.pi):
    pca = PCA(n_components=int(m), random_state=seed)
    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te)
    Ztr_phase, ntr = real_to_phase_embedding_linear(Z_tr, lam=lam)
    Zte_phase, nte = real_to_phase_embedding_linear(Z_te, lam=lam)
    return (Ztr_phase, ntr), (Zte_phase, nte)

def rp_transform_split(X_tr, X_te, m, seed=123, lam=np.pi, mode="gaussian"):
    rng = np.random.default_rng(seed)
    d = X_tr.shape[1]
    m = int(m)
    if mode == "gaussian":
        R = rng.normal(0.0, 1.0, size=(d, m))
    elif mode == "achlioptas":
        # Achlioptas sparse +/-1 with prob 1/6, else 0; scaled
        s = np.sqrt(3.0)
        U = rng.uniform(size=(d, m))
        R = np.zeros((d, m))
        R[U < 1/6] = +s
        R[(U >= 1/6) & (U < 2/6)] = -s
    else:
        raise ValueError("mode must be 'gaussian' or 'achlioptas'")

    Z_tr = X_tr @ R
    Z_te = X_te @ R
    Ztr_phase, ntr = real_to_phase_embedding_linear(Z_tr, lam=lam)
    Zte_phase, nte = real_to_phase_embedding_linear(Z_te, lam=lam)
    return (Ztr_phase, ntr), (Zte_phase, nte)
