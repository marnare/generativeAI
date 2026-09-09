import GBCcausal
from GBCcausal import GenTE, CausalIQN
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ganite import Ganite
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

grf = importr('grf')

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor


def simulate_causal_data(n_samples=500, 
                        n_features=4, treatment_type='observational', 
                        random_state=42, effect = "linear"):

    """Simulate Y = mu(X) + theta(X)*D + epsilon with known CATE."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    beta_mu = rng.standard_normal(n_features) * 0.5
    mu = X @ beta_mu

    if effect == "linear":
        theta_true = 1.0 + X[:, 0] + 0.5 * X[:, 1]
    elif effect == "non-linear":
        theta_true = 1.0 + X[:, 0]**2 + 0.5 * X[:, 1]**2
    elif effect == "interaction":
        theta_true = 1.0 + X[:, 0] * X[:, 1]
    elif effect == "non-linear interaction":
        theta_true = 1.0 + X[:, 0]**2 * X[:, 1]
    if treatment_type == 'rct':
        D = rng.binomial(1, 0.5, n_samples)
    else:
        logit = 0.3 * X[:, 0] - 0.2 * X[:, 1]
        if n_features >= 3:
            logit += 0.1 * X[:, 2]
        p = 1 / (1 + np.exp(-logit))
        D = rng.binomial(1, p, n_samples)

    Y = mu + theta_true * D + rng.standard_normal(n_samples) * 0.5
    return X, D, Y, theta_true




def run_mc_simulations(
    # ---- output ---------------------------------------------------
    results_dir='results_sims_mean',
    plot_show=True,
    save_plots=True,
    # ---- design ---------------------------------------------------
    effect_types=('linear', 'non-linear', 'interaction', 'non-linear interaction'),
    n_mc=100,
    n_samples=500,
    n_features=4,
    test_size=0.5,
    treatment_type='observational',
    seed=42,
    # ---- estimators -----------------------------------------------
    ganite_iterations=500,
    gente_kwargs=None,          # passed to CausalEnsembleModified
    gente_fit_kwargs=None,      # passed to .fit
    # ---- cosmetics -------------------------------------------------
    palette=('#8B0000', '#967BB6', '#E6A23C', '#4C787E'),
    figsize=(7, 5),
    verbose=True,
):
    """Monte Carlo comparison of GRF, GANITE and GenTE across effect designs.

    Returns a dict with per-design MSE series, summary statistics, and the
    last-replication predictions used for the boxplots.
    """
    effect_labels = {'linear': 'Linear',
                     'non-linear': 'Non-linear',
                     'interaction': 'Interaction',
                     'non-linear interaction': 'Non-linear interaction'}
    effect_filenames = {'linear': 'linear',
                        'non-linear': 'nonlinear',
                        'interaction': 'interaction',
                        'non-linear interaction': 'nonlinear_interaction'}

    gente_kwargs = {'model_cls': GBCcausal.CausalIQN,
                    'n_models': 3,
                    'device': 'auto',
                    'target': 'mean',
                    'hsz': 64,
                    'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 500, 'lr': 0.01, 'weight_decay': 1e-3,
                        'verbose': False,
                        **(gente_fit_kwargs or {})}

    os.makedirs(results_dir, exist_ok=True)
    effect_types = list(effect_types)
    mse_series, plot_data, summary = {}, {}, {}

    # ------------------------------------------------------------------
    # estimation
    # ------------------------------------------------------------------
    for effect in effect_types:
        if verbose:
            print(f"\n--- {effect.upper()} ({n_mc} MC replications) ---")
        mse_grf_list, mse_ganite_list, mse_gen_list = [], [], []
        last = None

        for rep in range(n_mc):
            rng_rep = seed + rep
            X, D, Y, tau_true = simulate_causal_data(
                n_samples=n_samples, n_features=n_features,
                treatment_type=treatment_type, effect=effect,
                random_state=rng_rep)
            (X_train, X_test, D_train, D_test,
             Y_train, Y_test, tau_train, tau_test) = train_test_split(
                X, D, Y, tau_true, test_size=test_size,
                random_state=rng_rep, stratify=D)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # --- GRF ---------------------------------------------------
            with localconverter(ro.default_converter + numpy2ri.converter):
                X_train_r = ro.r.matrix(X_train, nrow=X_train.shape[0],
                                        ncol=X_train.shape[1])
                D_train_r = ro.FloatVector(D_train)
                Y_train_r = ro.FloatVector(Y_train)
                X_test_r = ro.r.matrix(X_test, nrow=X_test.shape[0],
                                       ncol=X_test.shape[1])
                cf_sim = grf.causal_forest(X_train_r, Y_train_r, D_train_r,
                                           seed=rng_rep)
                pred_grf_sim = grf.predict_causal_forest(cf_sim, X_test_r)
                tau_grf_sim = np.array(
                    pred_grf_sim.rx2('predictions')
                    if hasattr(pred_grf_sim, 'rx2') else pred_grf_sim['predictions'])

            # --- GANITE ------------------------------------------------
            model_ganite = Ganite(X_train, D_train, Y_train,
                                  num_iterations=ganite_iterations)
            tau_ganite_sim = np.asarray(model_ganite(X_test).numpy()).flatten()

            # --- GenTE -------------------------------------------------
            kw = dict(gente_kwargs)
            hsz, nh = kw.pop('hsz'), kw.pop('nh')
            ens = GBCcausal.GenTE(
                model_kwargs={'xdim': X_train.shape[1], 'hsz': hsz, 'nh': nh},
                **kw)
            ens.fit(X_train, Y_train, D_train, **gente_fit_kwargs)
            tau_gen_sim = ens.estimate_qte(X_test).mean(axis=1).astype(np.float32)

            mse_grf_list.append(np.mean((tau_grf_sim - tau_test) ** 2))
            mse_ganite_list.append(np.mean((tau_ganite_sim - tau_test) ** 2))
            mse_gen_list.append(np.mean((tau_gen_sim - tau_test) ** 2))

            if rep == n_mc - 1:
                last = (tau_test, tau_grf_sim, tau_ganite_sim, tau_gen_sim)

        mse_series[effect] = {'GRF': np.array(mse_grf_list),
                              'GANITE': np.array(mse_ganite_list),
                              'GenTE': np.array(mse_gen_list)}
        summary[effect] = {k: (v.mean(), v.std())
                           for k, v in mse_series[effect].items()}
        plot_data[effect] = last

        if verbose:
            for k, (m, s) in summary[effect].items():
                print(f"{k:7s} MSE: {m:.2f} ± {s:.2f}")

    # ------------------------------------------------------------------
    # plots, on a shared scale across designs
    # ------------------------------------------------------------------
    all_vals = np.concatenate([np.concatenate(plot_data[e]) for e in effect_types])
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    pad = (y_max - y_min) * 0.05 + 0.01
    y_lim = (y_min - pad, y_max + pad)


    def _finish(fig, stem):
        fig.tight_layout()
        if save_plots:
            for ext in ('pdf', 'png'):
                fig.savefig(os.path.join(results_dir, f'{stem}.{ext}'),
                            dpi=300, bbox_inches='tight')
        if plot_show:
            display(fig)
        plt.close(fig)

    for effect in effect_types:
        tau_test_plot, tau_grf_plot, tau_ganite_plot, tau_gen_plot = plot_data[effect]
        m = summary[effect]
        fname = effect_filenames[effect]

        # ---- boxplot ---------------------------------------------------
        df_plot = pd.DataFrame({'True CATE': tau_test_plot,
                                'GRF': tau_grf_plot,
                                'GANITE': tau_ganite_plot,
                                'GenTE': tau_gen_plot}
                               ).melt(var_name='Method', value_name='CATE')
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df_plot, x='Method', y='CATE', hue='Method',
                    palette=list(palette), legend=False, ax=ax)
        ax.set_ylim(y_lim)
        ax.set_xlabel('Method')
        ax.set_ylabel('Conditional average treatment effect')
        ax.set_title(f"{effect_labels[effect]} (MSE: GRF {m['GRF'][0]:.2f}, "
                     f"GANITE {m['GANITE'][0]:.2f}, GenTE {m['GenTE'][0]:.2f})")
        _finish(fig, f'box_simulation_comparison_{fname}')

        # ---- scatter against the truth ---------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        for pred, colour, name in (
                (tau_grf_plot,    palette[1], 'GRF'),
                (tau_ganite_plot, palette[2], 'GANITE'),
                (tau_gen_plot,    palette[3], 'GenTE')):
            ax.scatter(tau_test_plot, pred, s=25, c=colour,
                       edgecolors='none', alpha=0.7,
                       label=f'{name} (MSE: {m[name][0]:.2f})')
        ax.plot(y_lim, y_lim, 'k--', linewidth=2, alpha=0.8, label='Perfect fit')
        ax.set_xlim(y_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('True CATE')
        ax.set_ylabel('Predicted CATE')
        ax.set_title(effect_labels[effect])
        ax.legend()
        _finish(fig, f'scatter_simulation_comparison_{fname}')

    return {'mse': mse_series, 'summary': summary,
            'plot_data': plot_data, 'y_lim': y_lim}





def run_mse_table(
    # ---- output ---------------------------------------------------
    results_dir='results_sims_mean',
    table_name='df_table_grf_gente_dml',
    save_table=True,
    # ---- design ---------------------------------------------------
    effect_types=('linear', 'non-linear', 'interaction', 'non-linear interaction'),
    n_samples_list=(200, 500, 1000),
    n_features_list=(2, 4, 8),
    n_mc=100,
    test_size=0.5,
    treatment_type='observational',
    seed=232,
    # ---- GRF ------------------------------------------------------
    grf_seed_per_rep=False,     # False reproduces the original fixed-seed forest
    # ---- GenTE ----------------------------------------------------
    gente_kwargs=None,          # passed to GenTE(...)
    gente_fit_kwargs=None,      # passed to .fit(...)
    # ---- DML ------------------------------------------------------
    dml_kwargs=None,            # passed to CausalForestDML(...)
    xgb_kwargs=None,            # outcome nuisance
    rf_kwargs=None,             # treatment nuisance
    verbose=True,
):
    """MSE of GRF, GenTE and DML-XGB across effect types, sample sizes and p.

    Returns the summary DataFrame; also writes it to
    ``results_dir/table_name.csv`` when ``save_table`` is True.
    """
    effect_labels = {'linear': 'Linear',
                     'non-linear': 'Non-linear',
                     'interaction': 'Interaction',
                     'non-linear interaction': 'Non-linear interaction'}

    gente_kwargs = {'model_cls': GBCcausal.CausalIQN, 'n_models': 3, 'device': 'auto',
                    'target': 'mean', 'hsz': 64, 'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 500, 'lr': 0.01, 'weight_decay': 1e-3,
                        'verbose': False, **(gente_fit_kwargs or {})}
    xgb_kwargs = {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
                  'subsample': 0.8, 'colsample_bytree': 0.8,
                  **(xgb_kwargs or {})}
    rf_kwargs = {'n_estimators': 200, 'min_samples_leaf': 5,
                 **(rf_kwargs or {})}
    dml_kwargs = {'discrete_treatment': True, 'n_estimators': 500,
                  'min_samples_leaf': 5, **(dml_kwargs or {})}

    os.makedirs(results_dir, exist_ok=True)
    effect_types = list(effect_types)
    rows = []

    for effect in effect_types:
        for n_samples in n_samples_list:
            for n_features in n_features_list:
                if verbose:
                    print(f"{effect:24s} S={n_samples:5d} p={n_features:2d} "
                          f"({n_mc} reps)", flush=True)

                mse_grf_list, mse_gen_list, mse_dml_list = [], [], []

                for rep in range(n_mc):
                    rng_rep = seed + rep
                    X, D, Y, tau_true = simulate_causal_data(
                        n_samples=n_samples, n_features=n_features,
                        treatment_type=treatment_type, effect=effect,
                        random_state=rng_rep)
                    (X_train, X_test, D_train, D_test,
                     Y_train, Y_test, tau_train, tau_test) = train_test_split(
                        X, D, Y, tau_true, test_size=test_size,
                        random_state=rng_rep, stratify=D)

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # ---------------- GRF ----------------
                    forest_seed = rng_rep if grf_seed_per_rep else seed
                    with localconverter(ro.default_converter
                                        + numpy2ri.converter):
                        X_train_r = ro.r.matrix(X_train, nrow=X_train.shape[0],
                                                ncol=X_train.shape[1])
                        D_train_r = ro.FloatVector(D_train)
                        Y_train_r = ro.FloatVector(Y_train)
                        X_test_r = ro.r.matrix(X_test, nrow=X_test.shape[0],
                                               ncol=X_test.shape[1])
                        cf = grf.causal_forest(X_train_r, Y_train_r, D_train_r,
                                               seed=forest_seed)
                        pred_grf = grf.predict_causal_forest(cf, X_test_r)
                        tau_grf = np.array(
                            pred_grf.rx2('predictions')
                            if hasattr(pred_grf, 'rx2')
                            else pred_grf['predictions'])

                    # ------------- GenTE -------------
                    kw = dict(gente_kwargs)
                    hsz, nh = kw.pop('hsz'), kw.pop('nh')
                    ens = GenTE(model_kwargs={'xdim': X_train.shape[1],
                                              'hsz': hsz, 'nh': nh}, **kw)
                    ens.fit(X_train, Y_train, D_train, **gente_fit_kwargs)
                    tau_gen = ens.estimate_qte(X_test).mean(axis=1)

                    # ------------- DML with XGBoost nuisances -------------
                    dml = CausalForestDML(
                        model_y=XGBRegressor(random_state=rng_rep,
                                             **xgb_kwargs),
                        model_t=RandomForestClassifier(random_state=rng_rep,
                                                       **rf_kwargs),
                        random_state=rng_rep,
                        **dml_kwargs)
                    dml.fit(Y_train, D_train, X=X_train)
                    tau_dml = dml.effect(X_test)

                    mse_grf_list.append(np.mean((tau_grf - tau_test) ** 2))
                    mse_gen_list.append(np.mean((tau_gen - tau_test) ** 2))
                    mse_dml_list.append(np.mean((tau_dml - tau_test) ** 2))

                grf_m, gen_m, dml_m = (np.mean(mse_grf_list),
                                       np.mean(mse_gen_list),
                                       np.mean(mse_dml_list))
                rows.append({
                    'effect': effect,
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'GRF MSE': grf_m,
                    'GRF MSE std': np.std(mse_grf_list),
                    'GenTE MSE': gen_m,
                    'GenTE MSE std': np.std(mse_gen_list),
                    'DML MSE': dml_m,
                    'DML MSE std': np.std(mse_dml_list),
                    'gain vs GRF (%)': 100.0 * (1.0 - gen_m / grf_m),
                    'gain vs DML (%)': 100.0 * (1.0 - gen_m / dml_m),
                })

    df_table = pd.DataFrame(rows)

    if save_table:
        path = os.path.join(results_dir, f'{table_name}.csv')
        df_table.to_csv(path, index=False)
        if verbose:
            print(f"\nsaved {path}")

    if verbose:
        for effect in effect_types:
            df_e = df_table[df_table['effect'] == effect]
            print(f"\n--- {effect_labels[effect].upper()} ---")
            for _, row in df_e.iterrows():
                print(f"  S={row['n_samples']:.0f}, p={row['n_features']:.0f}: "
                      f"GRF {row['GRF MSE']:.4f} ± {row['GRF MSE std']:.4f} | "
                      f"GenTE {row['GenTE MSE']:.4f} ± "
                      f"{row['GenTE MSE std']:.4f} | "
                      f"DML-XGB {row['DML MSE']:.4f} ± "
                      f"{row['DML MSE std']:.4f}")

    return df_table



def run_mse_table_ganite(
    # ---- output ---------------------------------------------------
    results_dir='results_sims_mean',
    table_name='df_table_ganite_gente',
    save_table=True,
    # ---- design ---------------------------------------------------
    effect_types=('linear', 'non-linear', 'interaction', 'non-linear interaction'),
    n_samples_list=(200, 500, 1000),
    n_features_list=(2, 4, 8),
    n_mc=100,
    test_size=0.5,
    treatment_type='observational',
    seed=232,
    # ---- estimators -----------------------------------------------
    ganite_iterations=500,
    gente_kwargs=None,          # passed to GenTE(...)
    gente_fit_kwargs=None,      # passed to .fit(...)
    verbose=True,
):
    """MSE of GANITE and GenTE across effect types, sample sizes and p.

    Returns the summary DataFrame; also writes it to
    ``results_dir/table_name.csv`` when ``save_table`` is True.
    """
    effect_labels = {'linear': 'Linear',
                     'non-linear': 'Non-linear',
                     'interaction': 'Interaction',
                     'non-linear interaction': 'Non-linear interaction'}

    gente_kwargs = {'model_cls': GBCcausal.CausalIQN, 'n_models': 3, 'device': 'auto',
                    'target': 'mean', 'hsz': 64, 'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 500, 'lr': 0.01, 'weight_decay': 1e-3,
                        'verbose': False, **(gente_fit_kwargs or {})}

    os.makedirs(results_dir, exist_ok=True)
    effect_types = list(effect_types)
    rows = []

    for effect in effect_types:
        for n_samples in n_samples_list:
            for n_features in n_features_list:
                if verbose:
                    print(f"{effect:24s} S={n_samples:5d} p={n_features:2d} "
                          f"({n_mc} reps)", flush=True)

                mse_ganite_list, mse_gen_list = [], []

                for rep in range(n_mc):
                    rng_rep = seed + rep
                    X, D, Y, tau_true = simulate_causal_data(
                        n_samples=n_samples, n_features=n_features,
                        treatment_type=treatment_type, effect=effect,
                        random_state=rng_rep)
                    (X_train, X_test, D_train, D_test,
                     Y_train, Y_test, tau_train, tau_test) = train_test_split(
                        X, D, Y, tau_true, test_size=test_size,
                        random_state=rng_rep, stratify=D)

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # ------------- GANITE -------------
                    model_ganite = Ganite(X_train, D_train, Y_train,
                                          num_iterations=ganite_iterations)
                    tau_ganite = np.asarray(
                        model_ganite(X_test).numpy()).flatten()

                    # ------------- GenTE -------------
                    kw = dict(gente_kwargs)
                    hsz, nh = kw.pop('hsz'), kw.pop('nh')
                    ens = GenTE(model_kwargs={'xdim': X_train.shape[1],
                                              'hsz': hsz, 'nh': nh}, **kw)
                    ens.fit(X_train, Y_train, D_train, **gente_fit_kwargs)
                    tau_gen = ens.estimate_qte(X_test).mean(axis=1)

                    mse_ganite_list.append(np.mean((tau_ganite - tau_test) ** 2))
                    mse_gen_list.append(np.mean((tau_gen - tau_test) ** 2))

                gan_m, gen_m = np.mean(mse_ganite_list), np.mean(mse_gen_list)
                rows.append({
                    'effect': effect,
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'GANITE MSE': gan_m,
                    'GANITE MSE std': np.std(mse_ganite_list),
                    'GenTE MSE': gen_m,
                    'GenTE MSE std': np.std(mse_gen_list),
                    'gain vs GANITE (%)': 100.0 * (1.0 - gen_m / gan_m),
                })

    df_table = pd.DataFrame(rows)

    if save_table:
        path = os.path.join(results_dir, f'{table_name}.csv')
        df_table.to_csv(path, index=False)
        if verbose:
            print(f"\nsaved {path}")

    if verbose:
        for effect in effect_types:
            df_e = df_table[df_table['effect'] == effect]
            print(f"\n--- {effect_labels[effect].upper()} ---")
            for _, row in df_e.iterrows():
                print(f"  S={row['n_samples']:.0f}, p={row['n_features']:.0f}: "
                      f"GANITE {row['GANITE MSE']:.4f} ± "
                      f"{row['GANITE MSE std']:.4f} | "
                      f"GenTE {row['GenTE MSE']:.4f} ± "
                      f"{row['GenTE MSE std']:.4f}")

    return df_table



def run_mse_table_sep_joint(
    # ---- output ---------------------------------------------------
    results_dir='results_sims_mean',
    table_name='df_table_sep_joint',
    save_table=True,
    # ---- design ---------------------------------------------------
    effect_types=('linear', 'non-linear', 'interaction', 'non-linear interaction'),
    n_samples_list=(200, 500, 1000),
    n_features_list=(2, 4, 8),
    n_mc=100,
    test_size=0.5,
    treatment_type='observational',
    seed=232,
    # ---- estimators -----------------------------------------------
    gente_kwargs=None,          # passed to GenTE(...)
    gente_fit_kwargs=None,      # joint: passed to .fit(...)
    sep_kwargs=None,            # separate: passed to .estimate_qte_separate(...)
    cate_n_mc=500,              # draws used by .estimate_cate(...)
    verbose=True,
):
    """MSE of GenTE fitted separately by arm (T-learner) versus jointly.

    Three estimators are scored:
      * ``GenTE Separate``           -- arms fitted separately, CATE by
                                        averaging theta(x, q) over the grid;
      * ``GenTE Joint Estimation``   -- joint fit, CATE by averaging
                                        theta(x, q) over the 19-point grid
                                        returned by ``estimate_qte``;
      * ``GenTE Joint MC``           -- the SAME joint fit, CATE from
                                        ``estimate_cate`` (``cate_n_mc``
                                        uniform draws per ensemble member).

    The two joint columns share a fit, so any difference between them is
    numerical approximation of the integral in the CATE definition, not a
    difference of estimator.

    Returns the summary DataFrame; also writes it to
    ``results_dir/table_name.csv`` when ``save_table`` is True.

    Note that ``estimate_qte_separate`` fits and predicts on the same
    covariates, so the separate estimator is scored on the evaluation fold it
    was fitted on, while the joint estimators are fitted on the training fold
    and scored out of sample.
    """
    effect_labels = {'linear': 'Linear',
                     'non-linear': 'Non-linear',
                     'interaction': 'Interaction',
                     'non-linear interaction': 'Non-linear interaction'}

    gente_kwargs = {'model_cls': GBCcausal.CausalIQN, 'n_models': 3, 'device': 'auto',
                    'target': 'mean', 'hsz': 64, 'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 500, 'lr': 0.01, 'weight_decay': 1e-3,
                        'verbose': False, **(gente_fit_kwargs or {})}
    sep_kwargs = {'epochs': 100, 'hdim': 64, 'nh': 32, **(sep_kwargs or {})}

    os.makedirs(results_dir, exist_ok=True)
    effect_types = list(effect_types)
    rows = []

    for effect in effect_types:
        for n_samples in n_samples_list:
            for n_features in n_features_list:
                if verbose:
                    print(f"{effect:24s} S={n_samples:5d} p={n_features:2d} "
                          f"({n_mc} reps)", flush=True)

                mse_sep_list, mse_joint_list, mse_joint_mc_list = [], [], []

                for rep in range(n_mc):
                    rng_rep = seed + rep
                    X, D, Y, tau_true = simulate_causal_data(
                        n_samples=n_samples, n_features=n_features,
                        treatment_type=treatment_type, effect=effect,
                        random_state=rng_rep)
                    (X_train, X_test, D_train, D_test,
                     Y_train, Y_test, tau_train, tau_test) = train_test_split(
                        X, D, Y, tau_true, test_size=test_size,
                        random_state=rng_rep, stratify=D)

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    kw = dict(gente_kwargs)
                    hsz, nh = kw.pop('hsz'), kw.pop('nh')
                    mk = {'xdim': X_train.shape[1], 'hsz': hsz, 'nh': nh}

                    # ---- GenTE, arms fitted separately (T-learner) ----
                    ens_sep = GenTE(model_kwargs=mk, **kw)
                    qte_sep = ens_sep.estimate_qte_separate(
                        X_test, Y_test, D_test, **sep_kwargs)
                    tau_sep = qte_sep.mean(axis=1)

                    # ---- GenTE, joint estimation (one fit, two summaries) ----
                    ens = GenTE(model_kwargs=mk, **kw)
                    ens.fit(X_train, Y_train, D_train, **gente_fit_kwargs)

                    # (i) grid average of theta(x, q)
                    tau_joint = ens.estimate_qte(X_test).mean(axis=1)
                    # (ii) Monte Carlo average from estimate_cate
                    tau_joint_mc = ens.estimate_cate(X_test, n_mc=cate_n_mc)['cate']

                    mse_sep_list.append(np.mean((tau_sep - tau_test) ** 2))
                    mse_joint_list.append(np.mean((tau_joint - tau_test) ** 2))
                    mse_joint_mc_list.append(np.mean((tau_joint_mc - tau_test) ** 2))

                sep_m = np.mean(mse_sep_list)
                joint_m = np.mean(mse_joint_list)
                joint_mc_m = np.mean(mse_joint_mc_list)
                rows.append({
                    'effect': effect,
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'GenTE Separate MSE': sep_m,
                    'GenTE Separate MSE std': np.std(mse_sep_list),
                    'GenTE Joint Estimation MSE': joint_m,
                    'GenTE Joint Estimation MSE std': np.std(mse_joint_list),
                    'GenTE Joint MC MSE': joint_mc_m,
                    'GenTE Joint MC MSE std': np.std(mse_joint_mc_list),
                    'gain of joint (%)': 100.0 * (1.0 - joint_m / sep_m),
                    'gain of joint MC (%)': 100.0 * (1.0 - joint_mc_m / sep_m),
                    'grid vs MC gap (%)': 100.0 * (joint_mc_m - joint_m) / joint_m,
                })

    df_table = pd.DataFrame(rows)

    if save_table:
        path = os.path.join(results_dir, f'{table_name}.csv')
        df_table.to_csv(path, index=False)
        if verbose:
            print(f"\nsaved {path}")

    if verbose:
        for effect in effect_types:
            df_e = df_table[df_table['effect'] == effect]
            print(f"\n--- {effect_labels[effect].upper()} ---")
            for _, row in df_e.iterrows():
                print(f"  S={row['n_samples']:.0f}, p={row['n_features']:.0f}: "
                      f"separate {row['GenTE Separate MSE']:.4f} ± "
                      f"{row['GenTE Separate MSE std']:.4f} | "
                      f"joint(grid) {row['GenTE Joint Estimation MSE']:.4f} ± "
                      f"{row['GenTE Joint Estimation MSE std']:.4f} | "
                      f"joint(MC) {row['GenTE Joint MC MSE']:.4f} ± "
                      f"{row['GenTE Joint MC MSE std']:.4f}")

    return df_table



def run_assignment_robustness(
    # ---- output ---------------------------------------------------
    results_dir='results_sims_mean',
    table_name='df_table_assignments',
    records_name='mse_records_assignments',
    plot_name='box_assignment_mse',
    save_table=True,
    save_plots=True,
    plot_show=True,
    # ---- design ---------------------------------------------------
    assignment_types=('rct', 'logit', 'nonlinear', 'trig'),
    effect='non-linear interaction',
    n_samples=500,
    n_features=20,
    n_mc=100,
    test_size=0.5,
    seed=232,
    # ---- estimator -------------------------------------------------
    gente_kwargs=None,          # passed to GenTE(...)
    gente_fit_kwargs=None,      # passed to .fit(...)
    # ---- cosmetics -------------------------------------------------
    palette=('#C9A7A7', '#B4C4D9', '#E6C9A8', '#A8C5B8'),
    figsize=(7, 5),
    verbose=True,
):
    """Out-of-sample MSE of GenTE under four treatment assignment mechanisms.

    The CATE is held fixed at ``effect`` and only the propensity varies, so
    differences across mechanisms are attributable to assignment alone.

    Returns
    -------
    dict with 'summary' (one row per mechanism) and 'records' (one row per
    replication, long form, used for the boxplot).
    """
    assignment_labels = {'rct': 'RCT',
                         'logit': 'Linear logit',
                         'nonlinear': 'Nonlinear logit',
                         'trig': 'Trigonometric'}

    gente_kwargs = {'model_cls': GBCcausal.CausalIQN, 'n_models': 3, 'device': 'auto',
                    'target': 'mean', 'hsz': 64, 'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 500, 'lr': 0.01, 'weight_decay': 1e-3,
                        'verbose': False, **(gente_fit_kwargs or {})}

    os.makedirs(results_dir, exist_ok=True)
    assignment_types = list(assignment_types)
    rows, records = [], []

    for assignment in assignment_types:
        if verbose:
            print(f"{assignment_labels[assignment]:18s} ({n_mc} reps)",
                  flush=True)

        mse_list = []
        for rep in range(n_mc):
            rng_rep = seed + rep
            X, D, Y, tau_true = simulate_causal_assignments(
                n_samples=n_samples, n_features=n_features,
                treatment_type=assignment, effect=effect,
                random_state=rng_rep)
            (X_train, X_test, D_train, D_test,
             Y_train, Y_test, tau_train, tau_test) = train_test_split(
                X, D, Y, tau_true, test_size=test_size,
                random_state=rng_rep, stratify=D)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            kw = dict(gente_kwargs)
            hsz, nh = kw.pop('hsz'), kw.pop('nh')
            ens = GenTE(model_kwargs={'xdim': X_train.shape[1],
                                      'hsz': hsz, 'nh': nh}, **kw)
            ens.fit(X_train, Y_train, D_train, **gente_fit_kwargs)
            tau_hat = ens.estimate_qte(X_test).mean(axis=1)

            mse = float(np.mean((tau_hat - tau_test) ** 2))
            mse_list.append(mse)
            records.append({'assignment': assignment,
                            'label': assignment_labels[assignment],
                            'rep': rep, 'MSE': mse})

        mse_arr = np.array(mse_list)
        rows.append({
            'assignment': assignment,
            'label': assignment_labels[assignment],
            'GenTE MSE': mse_arr.mean(),
            'GenTE MSE std': mse_arr.std(),
            'GenTE MSE median': np.median(mse_arr),
            'GenTE MSE q25': np.percentile(mse_arr, 25),
            'GenTE MSE q75': np.percentile(mse_arr, 75),
        })

        if verbose:
            print(f"  GenTE {mse_arr.mean():.4f} ± {mse_arr.std():.4f}")

    df_summary = pd.DataFrame(rows)
    df_records = pd.DataFrame(records)

    if save_table:
        p1 = os.path.join(results_dir, f'{table_name}.csv')
        p2 = os.path.join(results_dir, f'{records_name}.csv')
        df_summary.to_csv(p1, index=False)
        df_records.to_csv(p2, index=False)
        if verbose:
            print(f"\nsaved {p1}\nsaved {p2}")

    # ------------------------------------------------------------------
    # boxplot of the Monte Carlo MSE distribution by mechanism
    # ------------------------------------------------------------------
    df_records['Assignment'] = pd.Categorical(
        df_records['label'],
        categories=[assignment_labels[a] for a in assignment_types],
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df_records, x='Assignment', y='MSE', hue='Assignment',
                palette=list(palette)[:len(assignment_types)],
                legend=False, ax=ax)
    ax.set_xlabel('Assignment mechanism')
    ax.set_ylabel('Test-set MSE')
    fig.tight_layout()

    if save_plots:
        for ext in ('pdf', 'png'):
            path = os.path.join(results_dir, f'{plot_name}.{ext}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"saved {path}")
    if plot_show:
        display(fig)
    plt.close(fig)

    if verbose:
        print()
        for _, row in df_summary.iterrows():
            print(f"{row['label']}: GenTE {row['GenTE MSE']:.4f} ± "
                  f"{row['GenTE MSE std']:.4f}")

    return {'summary': df_summary, 'records': df_records}



def run_snr_robustness(
    # ---- output ---------------------------------------------------
    results_dir='results_sims_mean',
    table_name='df_table_snr',
    records_name='mse_records_snr',
    plot_name='line_snr_mse',
    save_table=True,
    save_plots=True,
    plot_show=True,
    # ---- design ---------------------------------------------------
    snr_values=(0.1, 0.5, 1, 2, 3),
    effect='non-linear interaction',
    n_samples=500,
    n_features=20,
    n_mc=100,
    test_size=0.5,
    seed=232,
    # ---- estimator -------------------------------------------------
    gente_kwargs=None,          # passed to GenTE(...)
    gente_fit_kwargs=None,      # passed to .fit(...)
    # ---- cosmetics -------------------------------------------------
    colour='#4C787E',
    figsize=(7, 5),
    verbose=True,
):
    """Out-of-sample MSE of GenTE as a function of the signal-to-noise ratio.

    Assignment is the linear logit of the main observational design and the
    CATE is held at ``effect``; only the residual scale varies.

    Returns
    -------
    dict with 'summary' (one row per SNR) and 'records' (one row per
    replication, long form).
    """
    gente_kwargs = {'model_cls': GBCcausal.CausalIQN, 'n_models': 3, 'device': 'auto',
                    'target': 'mean', 'hsz': 64, 'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 500, 'lr': 0.01, 'weight_decay': 1e-3,
                        'verbose': False, **(gente_fit_kwargs or {})}

    os.makedirs(results_dir, exist_ok=True)
    snr_values = list(snr_values)
    rows, records = [], []

    for snr in snr_values:
        if verbose:
            print(f"SNR = {snr:<5g} ({n_mc} reps)", flush=True)

        mse_list = []
        for rep in range(n_mc):
            rng_rep = seed + rep
            X, D, Y, tau_true = simulate_causal_snr(
                n_samples=n_samples, n_features=n_features, snr=snr,
                effect=effect, random_state=rng_rep)
            (X_train, X_test, D_train, D_test,
             Y_train, Y_test, tau_train, tau_test) = train_test_split(
                X, D, Y, tau_true, test_size=test_size,
                random_state=rng_rep, stratify=D)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            kw = dict(gente_kwargs)
            hsz, nh = kw.pop('hsz'), kw.pop('nh')
            ens = GenTE(model_kwargs={'xdim': X_train.shape[1],
                                      'hsz': hsz, 'nh': nh}, **kw)
            ens.fit(X_train, Y_train, D_train, **gente_fit_kwargs)
            tau_hat = ens.estimate_qte(X_test).mean(axis=1)

            mse = float(np.mean((tau_hat - tau_test) ** 2))
            mse_list.append(mse)
            records.append({'snr': snr, 'rep': rep, 'MSE': mse})

        mse_arr = np.array(mse_list)
        rows.append({
            'snr': snr,
            'GenTE MSE': mse_arr.mean(),
            'GenTE MSE std': mse_arr.std(),
            'GenTE MSE median': np.median(mse_arr),
            'GenTE MSE q25': np.percentile(mse_arr, 25),
            'GenTE MSE q75': np.percentile(mse_arr, 75),
        })

        if verbose:
            print(f"  GenTE {mse_arr.mean():.4f} ± {mse_arr.std():.4f}")

    df_summary = pd.DataFrame(rows)
    df_records = pd.DataFrame(records)

    if save_table:
        p1 = os.path.join(results_dir, f'{table_name}.csv')
        p2 = os.path.join(results_dir, f'{records_name}.csv')
        df_summary.to_csv(p1, index=False)
        df_records.to_csv(p2, index=False)
        if verbose:
            print(f"\nsaved {p1}\nsaved {p2}")

    # ------------------------------------------------------------------
    # mean MSE against SNR, with a +/- one standard deviation ribbon
    # ------------------------------------------------------------------
    s = df_summary['snr'].to_numpy(float)
    m = df_summary['GenTE MSE'].to_numpy(float)
    sd = df_summary['GenTE MSE std'].to_numpy(float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(s, m - sd, m + sd, color=colour, alpha=0.25, linewidth=0)
    ax.plot(s, m, color=colour, marker='o', ms=5, linewidth=1.8)
    ax.set_xlabel('Signal-to-noise ratio')
    ax.set_ylabel('Test-set MSE')
    ax.set_xticks(s)
    fig.tight_layout()

    if save_plots:
        for ext in ('pdf', 'png'):
            path = os.path.join(results_dir, f'{plot_name}.{ext}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"saved {path}")
    if plot_show:
        display(fig)
    plt.close(fig)

    if verbose:
        print()
        for _, row in df_summary.iterrows():
            print(f"SNR {row['snr']:g}: GenTE {row['GenTE MSE']:.4f} ± "
                  f"{row['GenTE MSE std']:.4f}")

    return {'summary': df_summary, 'records': df_records}




def simulate_causal_snr(n_samples=500,
                        n_features=4,
                        snr=1.0,
                        random_state=42,
                        effect='non-linear interaction'):
    """Simulate Y = mu(X) + theta(X)*D + epsilon with known CATE.

    Assignment is the linear logit of the main observational design.
    The residual scale is set so that

        snr = Var(mu(X) + theta(X) D) / Var(epsilon).
    """
    if snr <= 0:
        raise ValueError(f'snr must be positive, got {snr}')

    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    beta_mu = rng.standard_normal(n_features) * 0.5
    mu = X @ beta_mu

    if effect == 'linear':
        theta_true = 1.0 + X[:, 0] + 0.5 * X[:, 1]
    elif effect == 'non-linear':
        theta_true = 1.0 + X[:, 0]**2 + 0.5 * X[:, 1]**2
    elif effect == 'interaction':
        theta_true = 1.0 + X[:, 0] * X[:, 1]
    elif effect == 'non-linear interaction':
        theta_true = 1.0 + X[:, 0]**2 * X[:, 1]
    else:
        raise ValueError(f'Unknown effect: {effect}')

    index = 0.3 * X[:, 0] - 0.2 * X[:, 1]
    if n_features >= 3:
        index += 0.1 * X[:, 2]
    p = 1.0 / (1.0 + np.exp(-index))
    D = rng.binomial(1, p, n_samples)

    signal = mu + theta_true * D
    sigma = np.sqrt(np.var(signal) / snr)
    Y = signal + rng.standard_normal(n_samples) * sigma
    return X, D, Y, theta_true





def simulate_causal_assignments(n_samples=500,
                        n_features=4,
                        treatment_type='logit',
                        random_state=42,
                        effect='non-linear interaction'):
    """Simulate Y = mu(X) + theta(X)*D + epsilon with known CATE.

    treatment_type
        'rct'        : e(x) = 1/2
        'logit'      : linear logistic (same as the main observational design)
        'nonlinear'  : logistic of a quadratic-interaction index
        'trig'       : non-monotone propensity, 1/2 + (2/5) sin(x1) cos(x2)
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    beta_mu = rng.standard_normal(n_features) * 0.5
    mu = X @ beta_mu

    if effect == 'linear':
        theta_true = 1.0 + X[:, 0] + 0.5 * X[:, 1]
    elif effect == 'non-linear':
        theta_true = 1.0 + X[:, 0]**2 + 0.5 * X[:, 1]**2
    elif effect == 'interaction':
        theta_true = 1.0 + X[:, 0] * X[:, 1]
    elif effect == 'non-linear interaction':
        theta_true = 1.0 + X[:, 0]**2 * X[:, 1]
    else:
        raise ValueError(f'Unknown effect: {effect}')

    if treatment_type == 'rct':
        p = np.full(n_samples, 0.5)
    elif treatment_type in ('logit', 'observational'):
        index = 0.3 * X[:, 0] - 0.2 * X[:, 1]
        if n_features >= 3:
            index += 0.1 * X[:, 2]
        p = 1.0 / (1.0 + np.exp(-index))
    elif treatment_type == 'nonlinear':
        index = 0.3 * X[:, 0]**2 * X[:, 1]
        if n_features >= 3:
            index += 0.1 * X[:, 2]
        p = 1.0 / (1.0 + np.exp(-index))
    elif treatment_type == 'trig':
        p = 0.5 + 0.4 * np.sin(X[:, 0]) * np.cos(X[:, 1])
    else:
        raise ValueError(f'Unknown treatment_type: {treatment_type}')

    D = rng.binomial(1, p, n_samples)
    Y = mu + theta_true * D + rng.standard_normal(n_samples) * 0.5
    return X, D, Y, theta_true




from scipy.stats import norm

def simulate_regime(regime, n=2000, p=4, sigma0=0.5, sigma1=1.5, seed=7):
    """A: constant.  B: varies in x only.  C: varies in q only."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.normal(0, 0.5, size=p)
    mu = X @ beta

    theta_x = (1.0 + X[:, 0] + 0.5 * X[:, 1]) if regime == 'B' else np.ones(n)
    s1 = sigma1 if regime == 'C' else sigma0

    pi = 1 / (1 + np.exp(-(0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2])))
    D = rng.binomial(1, pi).astype(float)

    eps = np.where(D == 1, rng.normal(0, s1, n), rng.normal(0, sigma0, n))
    Y = mu + theta_x * D + eps
    return X, D, Y, theta_x, sigma0, s1


def true_qte(regime, theta_x_eval, q_grid, sigma0, s1):
    return theta_x_eval[:, None] + (s1 - sigma0) * norm.ppf(q_grid)[None, :]


def run_panels(regimes=('A', 'B', 'C'), n=2000, p=4, n_profiles=5,
               q_grid=None, seed=7, epochs=500):
    if q_grid is None:
        q_grid = (np.arange(19) + 0.5) / 19
    out = {}
    for reg in regimes:
        X, D, Y, theta_x, s0, s1 = simulate_regime(reg, n=n, p=p, seed=seed)

        # fixed evaluation profiles: sweep x1, hold the rest at 0
        X_eval = np.zeros((n_profiles, p))
        X_eval[:, 0] = np.linspace(-1.5, 1.5, n_profiles)
        theta_eval = (1 + X_eval[:, 0] + 0.5 * X_eval[:, 1]) if reg == 'B' \
                     else np.ones(n_profiles)

        sc = StandardScaler()
        X_s, X_eval_s = sc.fit_transform(X), sc.transform(X_eval)

        ens = GenTE(model_kwargs={'xdim': p, 'hsz': 64, 'nh': 32},
                    model_cls=GBCcausal.CausalIQN, n_models=3,
                    device='auto', target='mean')
        ens.fit(X_s, Y, D, epochs=epochs, lr=0.01, weight_decay=0, verbose=False)

        out[reg] = dict(
            q=q_grid,
            est=ens.estimate_qte(X_eval_s, quantiles=q_grid),
            truth=true_qte(reg, theta_eval, q_grid, s0, s1),
            theta_eval=theta_eval,
        )
    return out


def plot_panels(out, probit=True, n_profiles=None):
    regs = list(out)
    titles = {'A': 'No heterogeneity',
              'B': 'Covariate heterogeneity',
              'C': 'Quantile heterogeneity'}
    if n_profiles is None:
        n_profiles = out[regs[0]]['est'].shape[0]
    x1_eval = np.linspace(-1.5, 1.5, n_profiles)   # must match run_panels

    fig, axes = plt.subplots(1, len(regs), figsize=(4.1 * len(regs), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, reg in zip(axes, regs):
        d = out[reg]
        xs = norm.ppf(d['q']) if probit else d['q']
        for j in range(d['est'].shape[0]):
            c = plt.cm.viridis(j / max(d['est'].shape[0] - 1, 1))
            ax.plot(xs, d['est'][j], lw=1.7, color=c,
                    label=rf"$x_1={x1_eval[j]:+.2f}$")
            ax.plot(xs, d['truth'][j], ls='--', lw=1.1, color=c)
        ax.set_title(titles.get(reg, reg), fontsize=10)
        ax.set_xlabel(r"$\Phi^{-1}(q)$" if probit else r"$q$")
        ax.axhline(0, lw=0.5, color='0.7')

    axes[0].set_ylabel(r"$\hat\theta(x,q)$")
    # legend on the covariate-heterogeneity panel, where the profiles separate
    leg_ax = axes[regs.index('B')] if 'B' in regs else axes[0]
    leg_ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    return fig
