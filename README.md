# Generative Learner for Distributional Causal Effects

This repository contains the replication code. The code is adapted from ```https://github.com/VadimSokolov/gbc/blob/main/gbc/causal.py```.

## Replication Steps

1. Clone the repository:
   - `git clone https://github.com/yourusername/generativeAI.git`
   - `cd generativeAI`
  
 2. Install packages in the `requirements.txt`
 3. Run jupyter notebook `application_stars_mean_target.ipynb`
 4. Run `MC_simulations_mean_target.ipynb`


# Example 


```

"""
Toy example for GenTE (Generative Learner for Distributional Causal Effects).

Generates synthetic data with a KNOWN conditional average treatment effect
(the paper's "linear" design), fits GenTE, and checks the recovered CATE / ATE
/ QTE against ground truth.

Requirements:  `GBCcausal.py` (repo: src/) plus `gbc`
helper package, torch, numpy, scikit-learn.
Run:  python toy_gente.py
"""

import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.model_selection import train_test_split

# Make the repo's src/ importable (adjust if your layout differs).


import GBCcausal


# ----------------------------------------------------------------------
# 1. Synthetic data-generating process with a known CATE
#    Y = mu(X) + theta(X) * T + eps,   T ~ Bernoulli(pi(X))
# ----------------------------------------------------------------------
def make_data(n=1000, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))                      # X ~ N(0, I_p)

    beta_mu = rng.normal(0, 0.5, size=p)                 # baseline coefs
    mu = X @ beta_mu                                     # mu(X)

    # heterogeneous treatment effect (paper's "linear" design)
    theta = 1.0 + X[:, 0] + 0.5 * X[:, 1]                # true CATE

    # selection on observables: pi(X) = sigmoid(0.3 x1 - 0.2 x2 + 0.1 x3)
    logit = 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    pi = 1.0 / (1.0 + np.exp(-logit))
    T = rng.binomial(1, pi).astype(np.float32)

    eps = rng.normal(0, 0.5, size=n)
    Y = mu + theta * T + eps

    return (X.astype(np.float32),
            Y.astype(np.float32),
            T,
            theta.astype(np.float32))                    # theta = ground-truth CATE


X, Y, D, theta_true = make_data(n=1000, p=5, seed=0)

X_tr, X_te, Y_tr, Y_te, D_tr, D_te, theta_tr, theta_te = train_test_split(
    X, Y, D, theta_true, test_size=0.5, random_state=0
)


# GenTE parameters
gente_kwargs = {
    "model_cls": GBCcausal.CausalIQN,
    "n_models": 3,
    "device": "auto",
    "target": "mean",
    "hsz": 64,
    "nh": 32,
}
gente_fit_kwargs = {
    "epochs": 500,
    "lr": 0.01,
    "weight_decay": 1e-3,
    "verbose": False,
}
# Build GenTE
kw = dict(gente_kwargs)
hsz, nh = kw.pop("hsz"), kw.pop("nh")
ens = GBCcausal.GenTE(
    model_kwargs={
        "xdim": X_tr.shape[1],
        "hsz": hsz,
        "nh": nh,
    },
    **kw,
)
# Fit
ens.fit(
    X_tr,
    Y_tr,
    D_tr,
    **gente_fit_kwargs,
)
# Estimate CATE on test data
cate_out = ens.estimate_cate(X_te)
tau_gen_test = cate_out["cate"]
# ATE
ate_gen = cate_out["ate"]
# QTE
q_grid = np.linspace(0.05, 0.95, 19)
qte_gen = ens.estimate_qte(
    X_te,
    quantiles=q_grid,
)

# ======================================================================
# 3. Evaluate against ground truth
# ======================================================================
cate_mse = float(np.mean((tau_gen_test - theta_te) ** 2))
naive_mse = float(np.mean((theta_te.mean() - theta_te) ** 2))  # constant-ATE baseline
corr = float(np.corrcoef(tau_gen_test, theta_te)[0, 1])

print("\n================ GenTE toy example ================")
print(f"train / test sizes      : {len(X_tr)} / {len(X_te)}")
print(f"true  ATE               : {theta_te.mean(): .3f}")
print(f"GenTE ATE               : {ate_gen: .3f}")
print(f"CATE MSE (GenTE)        : {cate_mse: .3f}")
print(f"corr(estimate, truth)   : {corr: .3f}")
print(f"QTE grid shape          : {qte_gen.shape}  (n_test x n_quantiles)")
print(f"QTE mean over quantiles : {qte_gen.mean(axis=0).round(2)}")


```



