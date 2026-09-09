import torch
import numpy as np
import pandas as pd
import os
import random
import rpy2.robjects as ro
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import GBCcausal
from GBCcausal import GenTE, CausalIQN
from GBCcausal import GenTE, CausalIQN
import sys
sys.path.insert(0, 'src')
import GBCcausal
from GBCcausal import GenTE, CausalIQN
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
grf = importr('grf')

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.stats import gaussian_kde


# Set all seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set R seeds
ro.r('set.seed')(SEED)

def load_and_split_star_data(test_size=0.5, random_state=SEED):
    """
    Load and preprocess stars dataset.

    Covariates are stellar radius plus one-hot encoded colour and category.
    Treatment is a median split of temperature; the outcome is luminosity.
    Supergiants and hypergiants are dropped: in this catalogue they form a
    high-luminosity locus with little temperature contrast.

    Returns
    -------
    X_train, D_train, Y_train, X_test, D_test, Y_test : raw-scale arrays
    Y_train_std, Y_test_std : outcome standardized by the training moments
    y_loc, y_scale : those moments, for returning effects to the raw scale
    T_train, T_test : temperature in K, unstandardized.  It defines the
        treatment so it cannot be a covariate, but the Stefan--Boltzmann
        benchmark needs it as an outcome in the form (T / T_sun)^4.
    R_train, R_test : radius in solar units, unstandardized, for the r^2
        prefactor of the law.  X holds standardized radius and log radius.
    cat_train, cat_test : Star category labels, aligned with the split.
    """
    # Load from local CSV file
    data = pd.read_csv('Stars.csv')
    #data = data.loc[data['Star category'] != 'Hypergiant'].reset_index(drop=True)

    print("Dataset loaded successfully")
    print("Raw data shape:", data.shape)
    print("Available columns:", data.columns.tolist())
    print("After dropping hypergiants:", data.shape)

    # Catalogue is ordered by type; shuffle before the split so that order
    # cannot leak into train/test.  train_test_split shuffles again with
    # the same seed and stratifies on D.
    # data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

  
    # Covariates: radius and log radius plus stellar classification.
    # Raw radius is dominated by hypergiants after standardizing; log radius
    # keeps within-type size variation visible to the network.
    # Temperature defines the treatment, luminosity is the outcome, and
    # absolute magnitude is an affine function of log luminosity, so none
    # of the three may enter X.
    numerical_features = ['Radius (R/Ro)']
    log_radius = np.log(data['Radius (R/Ro)']).rename('log_Radius (R/Ro)')

    # Star colour is recorded inconsistently ('Blue White', 'Blue-white', ...);
    # normalise case, separators and stray whitespace before encoding.
    colour = (data['Star color'].str.strip().str.lower()
              .str.replace(r'[-_]', ' ', regex=True)
              .str.replace(r'\s+', ' ', regex=True))

    X_numerical = pd.concat(
        [data[numerical_features],
        pd.get_dummies(colour, prefix='color').astype(float),
        #pd.get_dummies(data['Star category'], prefix='cat').astype(float)
        ],
        axis=1,
    )
    
    # Create binary treatment based on temperature
    temperature_median = data['Temperature (K)'].median()
    D = (data['Temperature (K)'] > temperature_median).astype(int)
    
    # Outcome: Luminosity
    Y = data['Luminosity (L/Lo)']
    
    T = data['Temperature (K)']
    R = data['Radius (R/Ro)']
    cat = data['Star category']

    print("\nObservations by size (Star category):")
    print(cat.value_counts().sort_index().to_string())

    # Full-sample fit: CATE and the SB check describe this catalogue.
    # Names stay _train/_test so the estimation cell does not change; both
    # aliases are the same 240 stars when test_size is 0 or None.
    if test_size in (0, None):
        X_train = X_test = X_numerical
        D_train = D_test = D
        Y_train = Y_test = Y
        T_train = T_test = T
        R_train = R_test = R
        cat_train = cat_test = cat
    else:
        # Stratify by treatment
        (X_train, X_test, D_train, D_test, Y_train, Y_test,
         T_train, T_test, R_train, R_test,
         cat_train, cat_test) = train_test_split(
            X_numerical, D, Y, T, R, cat,
            test_size=test_size, random_state=random_state, stratify=D
        )
    
    # Optional scale (trees do not need it; the net does).  Always pass a
    # float ndarray to rpy2 / CausalEnsemble — StandardScaler was doing
    # that conversion as a side effect.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    D_train = np.asarray(D_train, dtype=float)
    D_test = np.asarray(D_test, dtype=float)


    # Standardize the outcome using training moments only.  Gradient-based
    # estimators need this because luminosity is of order 1e5; the transform is
    # affine, so effects return to the original scale on multiplication by
    # y_scale alone (the location cancels in a difference).
    Y_train = np.asarray(Y_train, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float)
    y_loc, y_scale = Y_train.mean(), Y_train.std()
    Y_train_std = (Y_train - y_loc) / y_scale
    Y_test_std = (Y_test - y_loc) / y_scale
    
    print("\nDataset Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("\nObservations by size, train:")
    print(pd.Series(cat_train).value_counts().sort_index().to_string())
    print("\nObservations by size, test:")
    print(pd.Series(cat_test).value_counts().sort_index().to_string())
    print(f"Features ({X_numerical.shape[1]}): {list(X_numerical.columns)}")
    print(f"Training treatment proportion: {D_train.mean():.3f}")
    print(f"Test treatment proportion: {D_test.mean():.3f}")
    print("\nTreatment definition:")
    print(f"Stars with temperature > {temperature_median:.2f}K are treated (hot stars)")
    print(f"Outcome: Luminosity (L/Lo)")
    print(f"Outcome standardization: centred at {y_loc:.4g}, divided by {y_scale:.4g}")
    print(f"Temperature returned unstandardized: "
          f"[{T_train.min():.0f}, {T_train.max():.0f}] K")
    
    return (X_train, D_train, Y_train, X_test, D_test, Y_test,
            Y_train_std, Y_test_std, y_loc, y_scale,
            np.asarray(T_train, dtype=float), np.asarray(T_test, dtype=float),
            np.asarray(R_train, dtype=float), np.asarray(R_test, dtype=float),
            np.asarray(cat_train), np.asarray(cat_test))



def load_and_split_star_data_covariates(test_size=0.5, random_state=SEED,
                             feature_set='full',
                             keep_categories=None):
    """
    Load and preprocess stars dataset.

    Treatment is a median split of temperature; the outcome is luminosity.

    feature_set : {'full', 'radius'}
        'full'   radius (level and log), colour, spectral class and category.
                 Colour and spectral class are temperature labels: spectral
                 class is the O--M temperature ordering and D is deterministic
                 within it for 235 of the 240 stars, so pi(x) is degenerate
                 almost everywhere under this set.
        'radius' radius in level and log only.  Radius is the covariate that
                 varies within temperature regimes, so pi(x) stays interior
                 for main-sequence stars, supergiants and hypergiants.
    keep_categories : None keeps all six classes.  Pass
        ('Main Sequence', 'Supergiant', 'Hypergiant') to restrict to the
        common-support subsample: every brown and red dwarf in this catalogue
        is cool and every white dwarf is hot, so pi(x) is 0 or 1 for those
        classes and their counterfactual arm is extrapolated.

    Returns
    -------
    X_train, D_train, Y_train, X_test, D_test, Y_test : raw-scale arrays
    Y_train_std, Y_test_std : outcome standardized by the training moments
    y_loc, y_scale : those moments, for returning effects to the raw scale
    T_train, T_test : temperature in K, unstandardized.  It defines the
        treatment so it cannot be a covariate, but the Stefan--Boltzmann
        benchmark needs it as an outcome in the form (T / T_sun)^4.
    R_train, R_test : radius in solar units, unstandardized, for the r^2
        prefactor of the law.  X holds standardized radius and log radius.
    cat_train, cat_test : Star category labels, aligned with the split.
    """
    data = pd.read_csv('Stars.csv')

    print("Dataset loaded successfully")
    print("Raw data shape:", data.shape)
    print("Available columns:", data.columns.tolist())

    if keep_categories is not None:
        before = len(data)
        data = data[data['Star category'].str.strip()
                    .isin(keep_categories)].reset_index(drop=True)
        print(f"Restricted to {list(keep_categories)}: "
              f"{len(data)} of {before} stars")

    # Covariates.  Raw radius is dominated by hypergiants after standardizing;
    # log radius keeps within-type size variation visible to the network.
    # Temperature defines the treatment, luminosity is the outcome, and
    # absolute magnitude is an affine function of log luminosity, so none of
    # the three may enter X.
    numerical_features = ['Radius (R/Ro)']
    log_radius = np.log(data['Radius (R/Ro)']).rename('log_Radius (R/Ro)')
    sq_radius = np.square(data['Radius (R/Ro)']).rename('sq_Radius (R/Ro)')
    sqrt_radius = np.sqrt(data['Radius (R/Ro)']).rename('sqrt_Radius (R/Ro)')
    cube_radius = np.power(data['Radius (R/Ro)'], 3).rename('cube_Radius (R/Ro)')


    if feature_set == 'radius':
        X_numerical = pd.concat([data[numerical_features], log_radius], axis=1)
    elif feature_set == 'full':
        # Star colour is recorded inconsistently ('Blue White', 'Blue-white',
        # 'blue white '); normalise case, separators and stray whitespace.
        colour = (data['Star color'].str.strip().str.lower()
                  .str.replace(r'[-_]', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True))
        X_numerical = pd.concat(
            [data[numerical_features],
             pd.get_dummies(colour, prefix='color').astype(float),
             pd.get_dummies(data['Spectral Class'].str.strip(),
                            prefix='spec').astype(float),
             pd.get_dummies(data['Star category'].str.strip(),
                            prefix='cat').astype(float)],
            axis=1,
        )
    else:
        raise ValueError("feature_set must be 'full' or 'radius', "
                         f"got {feature_set!r}")

    # Create binary treatment based on temperature
    temperature_median = data['Temperature (K)'].median()
    D = (data['Temperature (K)'] > temperature_median).astype(int)

    Y = data['Luminosity (L/Lo)']
    T = data['Temperature (K)']
    R = data['Radius (R/Ro)']
    cat = data['Star category'].str.strip()

    print("\nObservations and treated share by Star category:")
    print(pd.DataFrame({'cat': cat, 'D': D})
          .groupby('cat')['D'].agg(['size', 'mean']).to_string())

    # Full-sample fit: CATE and the SB check describe this catalogue.
    # Names stay _train/_test so the estimation cell does not change; both
    # aliases are the same stars when test_size is 0 or None.
    if test_size in (0, None):
        X_train = X_test = X_numerical
        D_train = D_test = D
        Y_train = Y_test = Y
        T_train = T_test = T
        R_train = R_test = R
        cat_train = cat_test = cat
    else:
        # Stratify by treatment
        (X_train, X_test, D_train, D_test, Y_train, Y_test,
         T_train, T_test, R_train, R_test,
         cat_train, cat_test) = train_test_split(
            X_numerical, D, Y, T, R, cat,
            test_size=test_size, random_state=random_state, stratify=D
        )

    # Optional scale (trees do not need it; the net does).  Always pass a
    # float ndarray to rpy2 / GenTE.
    scaler = StandardScaler()
    X_train = np.asarray(scaler.fit_transform(X_train), dtype=float)
    X_test = np.asarray(scaler.transform(X_test), dtype=float)
    D_train = np.asarray(D_train, dtype=float)
    D_test = np.asarray(D_test, dtype=float)

    # Standardize the outcome using training moments only.  Gradient-based
    # estimators need this because luminosity is of order 1e5; the transform is
    # affine, so effects return to the original scale on multiplication by
    # y_scale alone (the location cancels in a difference).
    Y_train = np.asarray(Y_train, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float)
    y_loc, y_scale = Y_train.mean(), Y_train.std()
    Y_train_std = (Y_train - y_loc) / y_scale
    Y_test_std = (Y_test - y_loc) / y_scale

    print("\nDataset Summary:")
    print(f"Feature set: {feature_set}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("\nObservations by size, train:")
    print(pd.Series(cat_train).value_counts().sort_index().to_string())
    print("\nObservations by size, test:")
    print(pd.Series(cat_test).value_counts().sort_index().to_string())
    print(f"Features ({X_numerical.shape[1]}): {list(X_numerical.columns)}")
    print(f"Training treatment proportion: {D_train.mean():.3f}")
    print(f"Test treatment proportion: {D_test.mean():.3f}")
    print("\nTreatment definition:")
    print(f"Stars with temperature > {temperature_median:.2f}K are treated "
          f"(hot stars)")
    print(f"Outcome: Luminosity (L/Lo)")
    print(f"Outcome standardization: centred at {y_loc:.4g}, "
          f"divided by {y_scale:.4g}")
    print(f"Temperature returned unstandardized: "
          f"[{T_train.min():.0f}, {T_train.max():.0f}] K")

    return (X_train, D_train, Y_train, X_test, D_test, Y_test,
            Y_train_std, Y_test_std, y_loc, y_scale,
            np.asarray(T_train, dtype=float), np.asarray(T_test, dtype=float),
            np.asarray(R_train, dtype=float), np.asarray(R_test, dtype=float),
            np.asarray(cat_train), np.asarray(cat_test))




def load_and_split_star_data_log(test_size=0.5, random_state=SEED):
    """
    Load and preprocess stars dataset.

    Covariates are stellar radius plus one-hot encoded colour and category.
    Treatment is a median split of temperature; the outcome is luminosity.
    Supergiants and hypergiants are dropped: in this catalogue they form a
    high-luminosity locus with little temperature contrast.

    Returns
    -------
    X_train, D_train, Y_train, X_test, D_test, Y_test : raw-scale arrays
    Y_train_std, Y_test_std : outcome standardized by the training moments
    y_loc, y_scale : those moments, for returning effects to the raw scale
    T_train, T_test : temperature in K, unstandardized.  It defines the
        treatment so it cannot be a covariate, but the Stefan--Boltzmann
        benchmark needs it as an outcome in the form (T / T_sun)^4.
    R_train, R_test : radius in solar units, unstandardized, for the r^2
        prefactor of the law.  X holds standardized radius and log radius.
    cat_train, cat_test : Star category labels, aligned with the split.
    """
    # Load from local CSV file
    data = pd.read_csv('Stars.csv')
    #data = data.loc[data['Star category'] != 'Hypergiant'].reset_index(drop=True)

    print("Dataset loaded successfully")
    print("Raw data shape:", data.shape)
    print("Available columns:", data.columns.tolist())
    print("After dropping hypergiants:", data.shape)

    # Catalogue is ordered by type; shuffle before the split so that order
    # cannot leak into train/test.  train_test_split shuffles again with
    # the same seed and stratifies on D.
    #data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

  
    # Covariates: radius and log radius plus stellar classification.
    # Raw radius is dominated by hypergiants after standardizing; log radius
    # keeps within-type size variation visible to the network.
    # Temperature defines the treatment, luminosity is the outcome, and
    # absolute magnitude is an affine function of log luminosity, so none
    # of the three may enter X.
    numerical_features = ['Radius (R/Ro)']
    log_radius = np.log(data['Radius (R/Ro)']).rename('log_Radius (R/Ro)')

    # Star colour is recorded inconsistently ('Blue White', 'Blue-white', ...);
    # normalise case, separators and stray whitespace before encoding.
    colour = (data['Star color'].str.strip().str.lower()
              .str.replace(r'[-_]', ' ', regex=True)
              .str.replace(r'\s+', ' ', regex=True))

    X_numerical = pd.concat(
        [data[numerical_features],
        pd.get_dummies(colour, prefix='color').astype(float),
        pd.get_dummies(data['Star category'], prefix='cat').astype(float)
        ],
        axis=1,
    )
    
    # Create binary treatment based on temperature
    temperature_median = data['Temperature (K)'].median()
    D = (data['Temperature (K)'] > temperature_median).astype(int)
    
    # Outcome: Luminosity
    Y = data['Luminosity (L/Lo)']
    
    T = data['Temperature (K)']
    R = data['Radius (R/Ro)']
    cat = data['Star category']

    print("\nObservations by size (Star category):")
    print(cat.value_counts().sort_index().to_string())

    # Full-sample fit: CATE and the SB check describe this catalogue.
    # Names stay _train/_test so the estimation cell does not change; both
    # aliases are the same 240 stars when test_size is 0 or None.
    if test_size in (0, None):
        X_train = X_test = X_numerical
        D_train = D_test = D
        Y_train = Y_test = Y
        T_train = T_test = T
        R_train = R_test = R
        cat_train = cat_test = cat
    else:
        # Stratify by treatment
        (X_train, X_test, D_train, D_test, Y_train, Y_test,
         T_train, T_test, R_train, R_test,
         cat_train, cat_test) = train_test_split(
            X_numerical, D, Y, T, R, cat,
            test_size=test_size, random_state=random_state, stratify=D
        )
    
    # Optional scale (trees do not need it; the net does).  Always pass a
    # float ndarray to rpy2 / CausalEnsemble — StandardScaler was doing
    # that conversion as a side effect.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    D_train = np.asarray(D_train, dtype=float)
    D_test = np.asarray(D_test, dtype=float)


    # Standardize the outcome using training moments only.  Gradient-based
    # estimators need this because luminosity is of order 1e5; the transform is
    # affine, so effects return to the original scale on multiplication by
    # y_scale alone (the location cancels in a difference).
    Y_train = np.asarray(Y_train, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float)
    y_loc, y_scale = Y_train.mean(), Y_train.std()
    #Y_train_std = (Y_train - y_loc) / y_scale
    #Y_test_std = (Y_test - y_loc) / y_scale
    Y_train_std = np.log(Y_train)
    Y_test_std = np.log(Y_test)
    
    print("\nDataset Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("\nObservations by size, train:")
    print(pd.Series(cat_train).value_counts().sort_index().to_string())
    print("\nObservations by size, test:")
    print(pd.Series(cat_test).value_counts().sort_index().to_string())
    print(f"Features ({X_numerical.shape[1]}): {list(X_numerical.columns)}")
    print(f"Training treatment proportion: {D_train.mean():.3f}")
    print(f"Test treatment proportion: {D_test.mean():.3f}")
    print("\nTreatment definition:")
    print(f"Stars with temperature > {temperature_median:.2f}K are treated (hot stars)")
    print(f"Outcome: Luminosity (L/Lo)")
    print(f"Outcome standardization: centred at {y_loc:.4g}, divided by {y_scale:.4g}")
    print(f"Temperature returned unstandardized: "
          f"[{T_train.min():.0f}, {T_train.max():.0f}] K")
    
    return (X_train, D_train, Y_train, X_test, D_test, Y_test,
            Y_train_std, Y_test_std, y_loc, y_scale,
            np.asarray(T_train, dtype=float), np.asarray(T_test, dtype=float),
            np.asarray(R_train, dtype=float), np.asarray(R_test, dtype=float),
            np.asarray(cat_train), np.asarray(cat_test))




def fit_stars_grf_gente(
    X_train, Y_train, D_train, X_test, D_test,
    # ---- estimation ------------------------------------------------
    seed=42,
    gente_kwargs=None,          # passed to GenTE(...)
    gente_fit_kwargs=None,      # passed to .fit(...)
    q_grid=None,                # grid for the fitted outcome surface
    verbose=True,
    learning_rate = 0.02, 
    wd = 0.003,
    target = 'mean',
):
    """Fit GRF and GenTE on a training fold and score the test fold.

    Parameters
    ----------
    X_train, Y_train, D_train : training covariates, outcome, treatment.
    X_test, D_test : test covariates and observed treatment; the latter is
        used only for the factual outcome surface ``L_hat``.

    Returns
    -------
    dict with both CATE estimates, the GenTE summary from ``estimate_cate``,
    the fitted outcome surface, the trained ensemble, and the quadrature grid.
    """
    gente_kwargs = {'model_cls': GBCcausal.CausalIQN, 'n_models': 3,
                    'device': 'auto', 'target': target, 'hsz': 64, 'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 3000, 'lr': learning_rate, 'weight_decay': wd,
                        'verbose': False, **(gente_fit_kwargs or {})}
    if q_grid is None:
        q_grid = (np.arange(19) + 0.5) / 19

    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    D_tr = np.asarray(D_train, dtype=float)
    D_te = np.asarray(D_test, dtype=float)
    Y_tr = np.asarray(Y_train, dtype=float)

    # ---- GRF -------------------------------------------------------
    if verbose:
        print("\nTraining GRF model...")
    with localconverter(ro.default_converter + numpy2ri.converter):
        X_train_r = ro.r.matrix(X_tr, nrow=X_tr.shape[0], ncol=X_tr.shape[1])
        D_train_r = ro.FloatVector(D_tr)
        Y_train_r = ro.FloatVector(Y_tr)
        X_test_r = ro.r.matrix(X_te, nrow=X_te.shape[0], ncol=X_te.shape[1])

        cf = grf.causal_forest(X_train_r, Y_train_r, D_train_r, seed=seed)
        pred_grf = grf.predict_causal_forest(cf, X_test_r)
        tau_grf_test = np.array(
            pred_grf.rx2('predictions') if hasattr(pred_grf, 'rx2')
            else pred_grf['predictions'])

    # ---- GenTE -----------------------------------------------------
    if verbose:
        print("Training GenTE...")
    kw = dict(gente_kwargs)
    hsz, nh = kw.pop('hsz'), kw.pop('nh')
    ens = GBCcausal.GenTE(
        model_kwargs={'xdim': X_tr.shape[1], 'hsz': hsz, 'nh': nh}, **kw)
    ens.fit(X_tr, Y_tr, D_tr, **gente_fit_kwargs)

    cate_out = ens.estimate_cate(X_te)
    tau_gen_test = cate_out['cate']

    # ---- fitted outcome surface: E[Y | X, D] at the observed treatment ----
    x_t = torch.tensor(X_te, dtype=torch.float32, device=ens.device)
    z_t = torch.tensor(D_te, dtype=torch.float32, device=ens.device)

    L_hat = np.zeros(len(X_te))
    for model in ens.models:
        model.eval()
        with torch.no_grad():
            for q in q_grid:
                f, _, _, _ = model(x_t, z_t, float(q))
                L_hat += f[:, 1].cpu().numpy()
    L_hat /= ens.n_models * len(q_grid)

    if verbose:
        print("\nTest Set Results:")
        print("GRF:")
        print(f"Average Treatment Effect: {np.mean(tau_grf_test):.2f}")
        print(f"Standard Deviation: {np.std(tau_grf_test):.2f}")

        print("\nGenTE (estimate_cate):")
        print(f"Average Treatment Effect: {cate_out['ate']:.2f}")
        print(f"ATE SE: {cate_out['ate_se']:.2f}")
        print(f"ATE 95% CI: ({cate_out['ate_ci'][0]:.2f}, "
              f"{cate_out['ate_ci'][1]:.2f})")
        print(f"CATE SD: {np.std(tau_gen_test):.2f}")
        print(f"Min: {np.min(tau_gen_test):.4f}  ")

    return {
        'tau_grf_test': tau_grf_test,
        'tau_gen_test': tau_gen_test,
        'cate_out': cate_out,
        'L_hat': L_hat,
        'ens': ens,
        'q_grid': q_grid,
    }



def plot_effect_density(
    theta,
    name='GenTE',
    stem='treatment_effects_density',
    results_dir='plots',
    colour='#4C787E',
    figsize=(8, 6),
    neg_tol=0.01,
    xlim=None,
    save_plots=True,
    plot_show=True,
    verbose=True,
):
    """Density of estimated treatment effects, with median and minimum marked.

    Parameters
    ----------
    theta : (n,) estimated effects.
    name : label used in the printed summary.
    stem : file stem; written as ``<results_dir>/<stem>.{pdf,png}``.
    neg_tol : an estimate counts as negative if it falls below ``-neg_tol``.
        Zero is the sign restriction the Stefan--Boltzmann law implies; a
        positive tolerance reports only violations larger than numerical noise.
    xlim : shared limits, for putting two estimators on the same axes.
    """
    theta = np.asarray(theta, float).ravel()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    median_effect = np.median(theta)
    min_effect = theta.min()
    neg = theta < -neg_tol


    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(data=theta, color=colour, fill=True, alpha=0.6,
                cut=0, bw_adjust=0.5, ax=ax, label='')

    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax.axvline(median_effect, color='gray', linestyle='--', alpha=0.6,
               label=f'Median = {median_effect:.2f}')
    ax.axvline(min_effect, color='gray', linestyle=':', alpha=0.6,
               label=f'Min = {min_effect:.2f}')
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel('The effect of surface temperature on stellar luminosity')
    ax.set_ylabel('Density')
    ax.set_title('')
    ax.legend()
    fig.tight_layout()

    if save_plots:
        for ext in ('pdf', 'png'):
            path = os.path.join(results_dir, f'{stem}.{ext}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"  saved {path}")
    if plot_show:
        display(fig)
    plt.close(fig)

    return {'median': median_effect, 'mean': float(theta.mean()),
            'min': float(min_effect), 'max': float(theta.max()),
            'negative_share': float(neg.mean())}





CAT_ORDER = ['brown dwarf', 'red dwarf', 'white dwarf',
             'main sequence', 'supergiant', 'hypergiant']
CAT_ABBREV = dict(zip(CAT_ORDER, ['BD', 'RD', 'WD', 'MS', 'S', 'H']))
CAT_COLOURS = ['#8C5A3C', '#C44E52', '#4C78A8',
               '#4C787E', '#7B68A6', '#D4A017']

def plot_star_heatmaps(
    T, L, tau, cat,
    t_cut=None,
    results_dir='plots',
    stems=('star_heatmap_TL', 'star_heatmap_Ttau'),
    tau_floor=1e-2,
    share_ylim=True,            # both panels on the outcome's range
    ylim=None,                  # explicit (lo, hi) in data units; overrides
    figsize=(5.5, 4.6),
    legend_loc='center right',
    legend_bbox=(1.0, 0.70),
    save_plots=True,
    plot_show=True,
    verbose=True,
):
    """Hertzsprung--Russell panels: luminosity and estimated CATE against
    temperature, with one soft density map per stellar class.

    Parameters
    ----------
    T, L, tau, cat : temperature, luminosity, estimated effect, and class
        label, all aligned on the same units (the test fold).
    t_cut : the threshold defining D.  Pass the full-sample median; defaults
        to the median of ``T`` if omitted.
    tau_floor : effects at or below zero are floored to this value so they can
        be shown on a log axis; the share floored is reported.
    share_ylim : put both panels on the luminosity range, so the effect can be
        read against the outcome it is a contrast of.  Points outside are
        reported and clipped by the axis.
    ylim : explicit (lo, hi) in data units; takes precedence over share_ylim.
    """
    T = np.asarray(T, dtype=float)
    L = np.asarray(L, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # class labels in the catalogue are inconsistently cased and spaced
    cat = (pd.Series(cat).astype(str).str.strip().str.lower()
           .str.replace(r'[-_]', ' ', regex=True)
           .str.replace(r'\s+', ' ', regex=True).to_numpy())

    missing = [c for c in CAT_ORDER if c not in set(cat)]
    if missing and verbose:
        print(f"warning: no rows for {missing}")

    if t_cut is None:
        t_cut = float(np.median(T))

    t_ticks = [2e3, 5e3, 1e4, 2e4, 4e4]
    t_ticklabels = [r'$2\times10^{3}$', r'$5\times10^{3}$',
                    r'$10^{4}$', r'$2\times10^{4}$', r'$4\times10^{4}$']
    t_lim = (min(T.min(), t_ticks[0]), max(T.max(), t_ticks[-1]))
    levels = [0.28, 0.48, 0.68, 0.88, 1.0]

    # shared vertical range, taken from the outcome
    # shared vertical range: the union of the outcome and the effect
    if ylim is not None:
        shared = (np.log10(ylim[0]), np.log10(ylim[1]))
    elif share_ylim:
        L_pos = L[np.isfinite(L) & (L > 0)]
        tau_f = np.where(np.isfinite(tau) & (tau > 0), tau, tau_floor)
        tau_pos = tau_f[tau_f > 0]
        lo = min(L_pos.min(), tau_pos.min()) * 0.5
        hi = max(L_pos.max(), tau_pos.max()) * 2
        shared = (np.log10(lo), np.log10(hi))
    else:
        shared = None

    def _soft_cmap(colour):
        r, g, b = to_rgb(colour)
        return LinearSegmentedColormap.from_list(
            'soft', [(r, g, b, 0.0), (r, g, b, 0.16), (r, g, b, 0.42)])

    def _class_kde(xi, yi, grid, shape, rng):
        if xi.size < 3:
            return None
        pts = np.vstack([xi, yi]) + rng.normal(0, 0.05, size=(2, xi.size))
        try:
            kde = gaussian_kde(pts, bw_method=0.5)
        except np.linalg.LinAlgError:
            return None
        Z = np.clip(kde(grid).reshape(shape), 0, None)
        return None if Z.max() <= 0 else Z / Z.max()

    def _panel(y, ylabel, stem, y_floor=None):
        y = np.asarray(y, dtype=float)
        if y_floor is not None:
            n_floored = int(np.sum(~(np.isfinite(y) & (y > 0))))
            if verbose and n_floored:
                print(f"{stem}: {n_floored} of {y.size} values "
                      f"({100 * n_floored / y.size:.1f}%) floored at {y_floor:g}")
            y = np.where(np.isfinite(y) & (y > 0), y, y_floor)

        y_pos = y[np.isfinite(y) & (y > 0)]
        if shared is not None:
            y_lo, y_hi = shared
            out = np.sum((np.log10(y_pos) < y_lo) | (np.log10(y_pos) > y_hi))
            if verbose and out:
                print(f"{stem}: {out} of {y_pos.size} values "
                      f"({100 * out / y_pos.size:.1f}%) outside the shared "
                      f"range [{10 ** y_lo:.3g}, {10 ** y_hi:.3g}]")
        else:
            y_lo, y_hi = np.log10(y_pos.min() * 0.5), np.log10(y_pos.max() * 2)

        xg = np.linspace(np.log10(t_lim[0]), np.log10(t_lim[1]), 120)
        yg = np.linspace(y_lo, y_hi, 120)
        Xg, Yg = np.meshgrid(xg, yg)
        grid = np.vstack([Xg.ravel(), Yg.ravel()])
        rng = np.random.default_rng(0)

        fig, ax = plt.subplots(figsize=figsize)
        for c, colour in zip(CAT_ORDER, CAT_COLOURS):
            m = (cat == c)
            if not m.any():
                continue
            xi = np.log10(T[m])
            yi = np.log10(np.where(y[m] > 0, y[m], y_pos.min() * 0.5))
            ok = np.isfinite(xi) & np.isfinite(yi)
            xi, yi = xi[ok], yi[ok]

            Z = _class_kde(xi, yi, grid, Xg.shape, rng)
            if Z is not None:
                ax.contourf(10 ** xg, 10 ** yg, Z, levels=levels,
                            cmap=_soft_cmap(colour), antialiased=True)
                ax.contour(10 ** xg, 10 ** yg, Z, levels=levels,
                           colors=[colour], linewidths=0.7, alpha=0.8)
            ax.scatter(10 ** xi, 10 ** yi, s=12, color=colour, alpha=0.8,
                       linewidths=0, label=CAT_ABBREV[c], zorder=3)

        ax.set_xscale('log')
        ax.set_xlim(*t_lim)
        ax.invert_xaxis()
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(t_ticklabels)
        ax.set_xlabel(r'Temperature (K)')
        ax.axvline(t_cut, ls='--', c='k', lw=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_yscale('log')
        ax.set_ylim(10 ** y_lo, 10 ** y_hi)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.legend(frameon=False, fontsize=7, loc=legend_loc,
                  bbox_to_anchor=legend_bbox, ncol=2)
        fig.tight_layout()

        if save_plots:
            os.makedirs(results_dir, exist_ok=True)
            for ext in ('pdf', 'png'):
                path = os.path.join(results_dir, f'{stem}.{ext}')
                fig.savefig(path, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"  saved {path}")
        if plot_show:
            display(fig)
        plt.close(fig)

    _panel(L, r'Luminosity ($\mathrm{L}/\mathrm{L}_{\odot}$)', stems[0])
    _panel(tau, r'CATE ($\widehat{\theta}(x)$)', stems[1], y_floor=tau_floor)


def plot_fit_scatter(Y_obs, L_hat, name='full covariates',
                     stem='Lhat_vs_Ytest', results_dir='plots',
                     colour='#4C787E', figsize=(6.2, 5.8), lim=None,
                     save_plots=True, plot_show=True, verbose=True):
    """Fitted against observed luminosity on log-log axes, with the 45 degree
    line.  Returns the limits used, so a second panel can share them."""
    y = np.asarray(Y_obs, dtype=float)
    f = np.asarray(L_hat, dtype=float)
    pos = (f > 0) & (y > 0)

    if verbose:
        r = np.corrcoef(np.log(f[pos]), np.log(y[pos]))[0, 1]
        print(f"{name}: n = {pos.sum()} of {y.size} plotted "
              f"({100 * (~pos).mean():.1f}% dropped, L_hat <= 0)")
        print(f"  corr(log L_hat, log Y) = {r:.4f},  R2 = {r ** 2:.4f}")
        print(f"  L_hat range {f[pos].min():.3g} to {f[pos].max():.3g}  |  "
              f"Y range {y[pos].min():.3g} to {y[pos].max():.3g}")

    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.labelsize': 16, 'xtick.labelsize': 14,
        'ytick.labelsize': 14, 'legend.fontsize': 14,
    })

    if lim is None:
        lim = [min(y[pos].min(), f[pos].min()),
               max(y[pos].max(), f[pos].max())]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y[pos], f[pos], s=22, alpha=0.75, color=colour,
               edgecolor='none')
    ax.plot(lim, lim, ls='--', c='k', lw=1, label='')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'observed $L$  ($L/L_\odot$)')
    ax.set_ylabel(r'$\hat L$  ($L/L_\odot$)')
    ax.set_title(name, fontsize=14)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    if save_plots:
        os.makedirs(results_dir, exist_ok=True)
        for ext in ('pdf', 'png'):
            path = os.path.join(results_dir, f'{stem}.{ext}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"  saved {path}")
    if plot_show:
        display(fig)
    plt.close(fig)
    return lim




import statsmodels.api as sm

T_SUN = 5772.0


from scipy import stats as _st


def sb_regression(L, R, T, name='fitted L', t_sun=T_SUN, alpha=0.05,
                  verbose=True):
    """OLS of log L on log R and log(T/T_sun); the law implies (0, 2, 4).

    Run it on the fitted luminosities to test the estimator, and on the
    observed luminosities as a control: the catalogue itself need not obey
    the law, and a rejection there is a property of the data rather than of
    the model.

    Returns a one-row DataFrame of coefficients, standard errors,
    (1 - alpha) confidence intervals, t-statistics and p-values against the
    law's values, whether each interval covers the law's value, the joint F
    test and R^2.
    """
    L = np.asarray(L, dtype=float)
    R = np.asarray(R, dtype=float)
    T = np.asarray(T, dtype=float)
    ok = (L > 0) & (R > 0) & (T > 0)

    df = pd.DataFrame({'log_L': np.log(L[ok]),
                       'log_R': np.log(R[ok]),
                       'log_T': np.log(T[ok] / t_sun)})
    fit = sm.OLS.from_formula('log_L ~ log_R + log_T', data=df).fit()

    law = {'Intercept': 0.0, 'log_R': 2.0, 'log_T': 4.0}
    joint = fit.f_test('Intercept = 0, log_R = 2, log_T = 4')
    ci = fit.conf_int(alpha=alpha)

    row = {'name': name, 'n': int(ok.sum()), 'dropped': int((~ok).sum()),
           'R2': fit.rsquared, 'df_resid': int(fit.df_resid),
           'F': float(np.squeeze(joint.fvalue)),
           'F_pvalue': float(np.squeeze(joint.pvalue))}
    for k, v in law.items():
        b, se = fit.params[k], fit.bse[k]
        lo, hi = ci.loc[k, 0], ci.loc[k, 1]
        t = (b - v) / se
        row[f'{k}_est'] = b
        row[f'{k}_se'] = se
        row[f'{k}_lo'] = lo
        row[f'{k}_hi'] = hi
        row[f'{k}_t'] = t
        row[f'{k}_p'] = 2 * _st.t.sf(abs(t), fit.df_resid)
        row[f'{k}_covers'] = bool(lo <= v <= hi)

    if verbose:
        print(f'--- {name}  (n = {row["n"]}'
              + (f', {row["dropped"]} dropped for non-positive values'
                 if row['dropped'] else '') + ')')
        print(f'{"term":<12}{"law":>6}{"est":>9}{"se":>8}'
              f'{f"{100*(1-alpha):.0f}% CI":>20}{"t vs law":>10}{"p":>8}{"":>4}')
        for k, v in law.items():
            mark = '' if row[f'{k}_covers'] else '  *'
            print(f'{k:<12}{v:>6.0f}{row[f"{k}_est"]:>9.3f}'
                  f'{row[f"{k}_se"]:>8.3f}'
                  f'{f"[{row[f'{k}_lo']:.3f}, {row[f'{k}_hi']:.3f}]":>20}'
                  f'{row[f"{k}_t"]:>10.2f}{row[f"{k}_p"]:>8.3f}{mark}')
        print(f'R2 = {row["R2"]:.3f}   joint F(3, {row["df_resid"]}) = '
              f'{row["F"]:.2f}, p = {row["F_pvalue"]:.3g}')
        print('  * interval excludes the value the law implies')
        print()

    return pd.DataFrame([row])



def run_star_propensity_logit(
    # ---- data: full covariate set ---------------------------------
    X_train, D_train, X_test, D_test, cat_test, R_test, T_test,
    # ---- data: radius-only covariate set --------------------------
    X_train_ra, D_train_ra, X_test_ra, D_test_ra,
    cat_test_ra, R_test_ra, T_test_ra,
    # ---- output ---------------------------------------------------
    labels=('full', 'radius'),
    results_dir='plots',
    table_name='df_table_propensity_logit',
    scores_name='propensity_scores_logit',
    plot_name='hist_propensity_logit',
    save_table=True,
    save_plots=True,
    plot_show=True,
    # ---- estimator -------------------------------------------------
    tol=0.05,
    C=1.0,
    use_cv=False,
    max_iter=5000,
    seed=SEED,
    # ---- cosmetics -------------------------------------------------
    colour='#6E7F80',
    figsize=(11, 4.2),
    bins=25,
    verbose=True,
):
    """Propensity scores from a logistic regression on the stars data.

    Fits P(D = 1 | X) by penalised logistic regression on the training
    half of each covariate set and predicts on the held-out half.  A
    star is counted as lacking common support when the fitted score
    lies below ``tol`` or above ``1 - tol``.

    The covariates are already standardised by the loader, so the
    penalty applies on a common scale.  With ``use_cv=True`` the
    penalty strength is chosen by five-fold cross-validation instead of
    being fixed at ``C``.

    Returns
    -------
    dict with 'summary' (one row per covariate set and category),
    'overall' (one row per covariate set), 'scores' (one row per
    held-out star) and 'models' (the fitted classifiers).
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    os.makedirs(results_dir, exist_ok=True)

    datasets = {
        labels[0]: (X_train, D_train, X_test, D_test,
                    cat_test, R_test, T_test),
        labels[1]: (X_train_ra, D_train_ra, X_test_ra, D_test_ra,
                    cat_test_ra, R_test_ra, T_test_ra),
    }

    rows, overall, scores, models = [], [], [], {}

    for name, (Xtr, Dtr, Xte, Dte, cate, Rte, Tte) in datasets.items():
        if verbose:
            print(f"\n=== {name} "
                  f"({Xtr.shape[1]} features) ===", flush=True)

        if use_cv:
            clf = LogisticRegressionCV(cv=5, max_iter=max_iter,
                                       random_state=seed)
        else:
            clf = LogisticRegression(C=C, max_iter=max_iter,
                                     random_state=seed)
        clf.fit(Xtr, np.asarray(Dtr).ravel())
        models[name] = clf

        pi_hat = clf.predict_proba(Xte)[:, 1]
        outside = (pi_hat < tol) | (pi_hat > 1 - tol)
        acc_in = clf.score(Xtr, np.asarray(Dtr).ravel())
        acc_out = clf.score(Xte, np.asarray(Dte).ravel())

        for s in range(len(pi_hat)):
            scores.append({'covariates': name,
                           'category': cate[s],
                           'D': float(Dte[s]),
                           'radius': float(Rte[s]),
                           'temperature': float(Tte[s]),
                           'pi_hat': float(pi_hat[s]),
                           'no_support': bool(outside[s])})

        df_s = pd.DataFrame({'cat': cate, 'D': Dte,
                             'pi': pi_hat, 'out': outside})
        for cat, sub in df_s.groupby('cat'):
            rows.append({
                'covariates': name,
                'category': cat,
                'n': len(sub),
                'share treated': sub['D'].mean(),
                'pi min': sub['pi'].min(),
                'pi median': sub['pi'].median(),
                'pi max': sub['pi'].max(),
                f'share outside [{tol:g}, {1 - tol:g}]': sub['out'].mean(),
            })

        overall.append({
            'covariates': name,
            'n': len(pi_hat),
            'share treated': float(np.mean(Dte)),
            'accuracy train': acc_in,
            'accuracy test': acc_out,
            'pi min': float(pi_hat.min()),
            'pi median': float(np.median(pi_hat)),
            'pi max': float(pi_hat.max()),
            'n outside': int(outside.sum()),
            'share outside': float(outside.mean()),
        })

        if verbose:
            print(f"  accuracy {acc_in:.3f} train, {acc_out:.3f} test")
            print(f"  pi in [{pi_hat.min():.3f}, {pi_hat.max():.3f}], "
                  f"{outside.sum()} of {len(pi_hat)} stars "
                  f"({outside.mean():.1%}) outside "
                  f"[{tol:g}, {1 - tol:g}]")

    df_summary = pd.DataFrame(rows)
    df_overall = pd.DataFrame(overall)
    df_scores = pd.DataFrame(scores)

    if save_table:
        p1 = os.path.join(results_dir, f'{table_name}.csv')
        p2 = os.path.join(results_dir, f'{table_name}_overall.csv')
        p3 = os.path.join(results_dir, f'{scores_name}.csv')
        df_summary.to_csv(p1, index=False)
        df_overall.to_csv(p2, index=False)
        df_scores.to_csv(p3, index=False)
        if verbose:
            print(f"\nsaved {p1}\nsaved {p2}\nsaved {p3}")

    # ------------------------------------------------------------------
    # one figure per covariate set, on a common vertical scale
    # ------------------------------------------------------------------
    counts_max = 0
    for name in datasets:
        pi_hat = df_scores.loc[df_scores['covariates'] == name,
                               'pi_hat'].to_numpy()
        counts, _ = np.histogram(pi_hat, bins=bins, range=(0, 1))
        counts_max = max(counts_max, counts.max())

    for name in datasets:
        pi_hat = df_scores.loc[df_scores['covariates'] == name,
                               'pi_hat'].to_numpy()

        fig, ax = plt.subplots(figsize=figsize)
        ax.axvspan(0, tol, color='0.88', zorder=0)
        ax.axvspan(1 - tol, 1, color='0.88', zorder=0)
        ax.hist(pi_hat, bins=bins, range=(0, 1), color=colour,
                alpha=0.85, edgecolor='white', linewidth=0.5, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, counts_max * 1.08)
        ax.set_xlabel(r'$\hat{\pi}(x)$')
        ax.set_ylabel('Held-out stars')
        fig.tight_layout()

        if save_plots:
            for ext in ('pdf', 'png'):
                path = os.path.join(results_dir,
                                    f'{plot_name}_{name}.{ext}')
                fig.savefig(path, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"saved {path}")
        if plot_show:
            display(fig)
        plt.close(fig)

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
        print(df_overall.to_string(index=False))
        print()
        print(df_summary.to_string(index=False))

    return {'summary': df_summary, 'overall': df_overall,
            'scores': df_scores, 'models': models}






def fit_stars_gente_qte(
    X_train, Y_train, D_train, X_test, D_test,
    # ---- estimation ------------------------------------------------
    gente_kwargs=None,          # passed to GenTE(...)
    gente_fit_kwargs=None,      # passed to .fit(...)
    q_grid=None,                # grid for the QTE curve and outcome surface
    verbose=True,
    learning_rate=0.02,
    wd=0.003,
    target='mean',
):
    """Fit GenTE on a training fold and return quantile-indexed effects.

    Parameters
    ----------
    X_train, Y_train, D_train : training covariates, outcome, treatment.
    X_test, D_test : test covariates and observed treatment; the latter is
        used only for the factual outcome surface ``L_hat``.
    q_grid : quantile levels at which theta(x, q) is evaluated. Defaults to
        19 midpoints of equal subintervals of (0, 1).

    Returns
    -------
    dict with the QTE matrix (n_test x len(q_grid)), the CATE implied by
    averaging it over q, the ``estimate_cate`` summary, the fitted outcome
    surface, the trained ensemble, and the grid.
    """
    gente_kwargs = {'model_cls': GBCcausal.CausalIQN, 'n_models': 3,
                    'device': 'auto', 'target': target, 'hsz': 64, 'nh': 32,
                    **(gente_kwargs or {})}
    gente_fit_kwargs = {'epochs': 3000, 'lr': learning_rate, 'weight_decay': wd,
                        'verbose': False, **(gente_fit_kwargs or {})}
    if q_grid is None:
        q_grid = (np.arange(19) + 0.5) / 19
    q_grid = np.asarray(q_grid, dtype=float)

    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    D_tr = np.asarray(D_train, dtype=float)
    D_te = np.asarray(D_test, dtype=float)
    Y_tr = np.asarray(Y_train, dtype=float)

    # ---- GenTE -----------------------------------------------------
    if verbose:
        print("Training GenTE...")
    kw = dict(gente_kwargs)
    hsz, nh = kw.pop('hsz'), kw.pop('nh')
    ens = GBCcausal.GenTE(
        model_kwargs={'xdim': X_tr.shape[1], 'hsz': hsz, 'nh': nh}, **kw)
    ens.fit(X_tr, Y_tr, D_tr, **gente_fit_kwargs)

    # ---- quantile treatment effects: theta(x, q) on the grid -------
    # rows are stars, columns are quantile levels; already averaged over
    # the K ensemble members inside estimate_qte
    qte_gen_test = ens.estimate_qte(X_te, quantiles=q_grid)

    # CATE implied by integrating the curve over q (Monte Carlo version
    # from estimate_cate is returned alongside for comparison)
    tau_gen_test = qte_gen_test.mean(axis=1)
    cate_out = ens.estimate_cate(X_te)



    return {
        'qte_gen_test': qte_gen_test,   # theta(x, q), n_test x len(q_grid)
        'tau_gen_test': tau_gen_test,   # CATE = mean over q
        'cate_out': cate_out,
        'ens': ens,
        'q_grid': q_grid,
    }

