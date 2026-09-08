# Generative AI for Validating Physics Laws

This repository contains the replication code.

## Replication Steps

1. Clone the repository:
   - `git clone https://github.com/yourusername/generativeAI.git`
   - `cd generativeAI`
  
 2. Install packages in the `requirements.txt`
 3. Run jupyter notebook `application_stars_mean_target.ipynb`
 4. Run `MC_simulations_mean_target.ipynb`


# Example 


```

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

```



