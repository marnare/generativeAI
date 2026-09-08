"""Causal inference via GBC: CATE/ATE estimation.

QuantNet architecture with propensity score embedding, treatment effect module,
and quantile-indexed output.

Two target specifications are supported:

target="median"
    Original specification:
        mu     = mu(X, pi_embed)
        output = mu(X) + te(X, tau) * Z
    The L1 location anchor remains active and CATE is summarized by the median
    over quantile-indexed treatment effects.

target="mean"
    Distributional/CATE specification:
        mu     = mu(X, pi_embed, tau)
        output = mu(X, tau) + te(X, tau) * Z
    The L1 location anchor is switched off.  The baseline therefore learns the
    control conditional quantile function through the pinball loss, while
    mu + te learns the treated conditional quantile function.  CATE is obtained
    by averaging te(X, tau) over tau.

Notes
-----
``z`` must be a float tensor (0.0 / 1.0) for BCELoss compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from gbc.loss import _sample_quantile_pair, composite_loss
from gbc.iqn import cosine_embed, train_iqn
from gbc.utils import get_device




# ============================================================
# Helpers
# ============================================================

def _paired_forward(model, x, z, tau, tau_other):
    """Evaluate two quantile levels with the same stochastic realization.

    This lets the crossing penalty compare quantile levels without adding
    differences caused by independent dropout masks.
    """
    cpu_before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(x.device) if x.is_cuda else None

    first = model(x, z, tau)

    cpu_after = torch.random.get_rng_state()
    cuda_after = torch.cuda.get_rng_state(x.device) if x.is_cuda else None

    torch.random.set_rng_state(cpu_before)
    if cuda_before is not None:
        torch.cuda.set_rng_state(cuda_before, x.device)

    try:
        second = model(x, z, tau_other)
    finally:
        torch.random.set_rng_state(cpu_after)
        if cuda_after is not None:
            torch.cuda.set_rng_state(cuda_after, x.device)

    return first, second


# ============================================================
# CausalIQN
# ============================================================

class CausalIQN(nn.Module):
    """Quantile neural network for causal inference.

    Parameters
    ----------
    xdim : int
        Number of covariates.
    hsz : int
        Hidden dimension.
    nh : int
        Number of cosine frequencies.
    target : {"median", "mean"}
        "median" preserves the original specification.  "mean" makes the
        baseline quantile-specific, switches off the L1 location anchor, and
        estimates CATE by averaging the quantile-specific treatment effects.
    """

    def __init__(self, xdim=1, hsz=32, nh=32, target="median"):
        super().__init__()
        if target not in {"mean", "median"}:
            raise ValueError("target must be 'mean' or 'median'")

        self.target = target
        self.nh = nh
        pisz, lw = 8, 16

        # Propensity network
        self.pi = nn.Sequential(
            nn.Linear(xdim, 16), nn.ReLU(),
            nn.Linear(16, pisz),
        )
        self.pi1 = nn.Sequential(
            nn.Linear(pisz, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Baseline network
        self.mu = nn.Sequential(
            nn.Linear(xdim + pisz, lw), nn.ReLU(),
            nn.Linear(lw, lw), nn.ReLU(),
            nn.Linear(lw, hsz),
        )
        self.mu1 = nn.Sequential(
            nn.Linear(hsz, lw), nn.ReLU(),
            nn.Linear(lw, lw), nn.ReLU(),
            nn.Linear(lw, 1),
        )

        # Treatment-effect network
        self.te = nn.Sequential(
            nn.Linear(xdim, lw), nn.ReLU(),
            nn.Linear(lw, lw), nn.ReLU(),
            nn.Linear(lw, hsz),
        )
        self.te1 = nn.Sequential(
            nn.Linear(hsz, lw), nn.ReLU(),
            nn.Linear(lw, 2),
        )

        # Quantile embedding
        self.tau_embed = nn.Sequential(nn.Linear(nh, hsz), nn.ReLU())

    def forward(self, x, z, tau):
        """Forward pass.

        Returns
        -------
        y_pred : two-column prediction used by the composite loss
        pi_logits : propensity logits
        te : two-column treatment-effect output
        mu : baseline output
        """
        tau_e = self.tau_embed(
            cosine_embed(tau, self.nh, device=x.device, dtype=x.dtype)
        )

        pi = self.pi(x)
        pi1 = self.pi1(pi)

        mu_h = self.mu(torch.cat((x, pi), dim=1))

        # ONLY change when target="mean": make baseline quantile-specific
        mu = self.mu1(tau_e * mu_h) if self.target == "mean" else self.mu1(mu_h)

        te = self.te1(tau_e * self.te(x))
        f = mu + te * z.view(-1, 1)

        return f, pi1, te, mu

    def loss_fn(self, x, y, z, w=(0.3, 0.1, 0.6)):
        """Composite outcome loss plus propensity BCE."""
        tau, tau_other = _sample_quantile_pair()
        first, second = _paired_forward(self, x, z, tau, tau_other)

        f, pi_logit, _, _ = first
        f_other = second[0]

        pi_loss = nn.functional.binary_cross_entropy_with_logits(
            pi_logit.view(-1), z.float()
        )

        # ONLY change when target="mean": switch off the L1/MAE location anchor
        loss_weights = (0.0, w[1], w[2]) if self.target == "mean" else w

        outcome_loss = composite_loss(
            y, f, tau, loss_weights, f_other=f_other, tau_other=tau_other
        )
        return outcome_loss + pi_loss

    def fit(self, x, y, z, epochs=3000, w=(0.3, 0.1, 0.6), lr=5e-4, wd=3e-3):
        """Train the causal IQN."""
        opt = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            loss = self.loss_fn(x, y, z, w)
            loss.backward()
            opt.step()

    def estimate_cate(self, x, z, n_mc=500):
        """Estimate CATE over quantile levels.

        For target="mean" the CATE is the mean over quantile levels; for
        target="median" the original median summary is retained.
        """
        n = x.shape[0]
        samples = torch.zeros((n, n_mc), device=x.device)

        with torch.no_grad():
            for i in range(n_mc):
                tau = torch.rand(1).item()
                _, _, te, _ = self(x, z, tau)
                samples[:, i] = te[:, 1]

        # CATE = integral over q for target="mean"
        if self.target == "mean":
            cate = samples.mean(dim=1)
        else:
            cate = samples.median(dim=1).values

        return cate.cpu().numpy(), samples.cpu().numpy()


# ============================================================
# CausalIQNv2
# ============================================================

class CausalIQNv2(nn.Module):
    r"""Causal IQN with additive quantile--covariate interaction.

    The treatment-effect network concatenates covariates and quantile features
    rather than multiplying them.  When target="mean" the control baseline also
    receives the quantile features, so that

        mu(x, q)             -> Q_0(q | x)
        mu(x, q) + te(x, q)  -> Q_1(q | x)

    and therefore te(x, q) -> Q_1(q | x) - Q_0(q | x).

    Parameters
    ----------
    xdim : int
        Number of covariates.
    hdim : int
        Hidden layer width.
    nh : int
        Number of cosine frequencies.
    pidim : int
        Dimension of propensity embedding.
    dropout : float
        Dropout probability.
    target : {"median", "mean"}
        Target specification.
    """

    def __init__(self, xdim, hdim=64, nh=32, pidim=16, dropout=0.1,
                 target="median"):
        super().__init__()
        if target not in {"mean", "median"}:
            raise ValueError("target must be 'mean' or 'median'")

        self.target = target
        self.nh = nh

        # Propensity-score network
        self.pi_net = nn.Sequential(
            nn.Linear(xdim, hdim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hdim, hdim // 2), nn.ReLU(),
            nn.Linear(hdim // 2, pidim),
        )
        self.pi_head = nn.Linear(pidim, 1)

        # Baseline: [x, pi_embed] and, when target="mean", the q features too
        mu_in = xdim + pidim + nh if target == "mean" else xdim + pidim
        self.mu_net = nn.Sequential(
            nn.Linear(mu_in, hdim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, 1),
        )

        # Treatment-effect network
        self.te_net = nn.Sequential(
            nn.Linear(xdim + nh, hdim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hdim, hdim), nn.ReLU(),
        )
        self.te_head = nn.Sequential(
            nn.Linear(hdim + xdim, hdim // 2), nn.ReLU(),
            nn.Linear(hdim // 2, 2),
        )

    def forward(self, x, z, tau):
        """Forward pass."""
        pi_embed = self.pi_net(x)
        pi_logit = self.pi_head(pi_embed)

        q_feat = cosine_embed(tau, self.nh, device=x.device, dtype=x.dtype)
        q_feat = q_feat.unsqueeze(0).expand(x.shape[0], -1)

        if self.target == "mean":
            mu_input = torch.cat([x, pi_embed, q_feat], dim=1)
        else:
            mu_input = torch.cat([x, pi_embed], dim=1)
        mu = self.mu_net(mu_input)

        te_hidden = self.te_net(torch.cat([x, q_feat], dim=1))
        te = self.te_head(torch.cat([te_hidden, x], dim=1))

        f = mu + te * z.view(-1, 1)

        return f, pi_logit, te, mu

    def loss_fn(self, x, y, z, w=(0.3, 0.1, 0.6)):
        """Composite outcome loss plus propensity BCE."""
        tau, tau_other = _sample_quantile_pair()
        first, second = _paired_forward(self, x, z, tau, tau_other)

        f, pi_logit, _, _ = first
        f_other = second[0]

        pi_loss = nn.functional.binary_cross_entropy_with_logits(
            pi_logit.view(-1), z.float()
        )

        # target="mean": disable the MAE/L1 anchor
        loss_weights = (0.0, w[1], w[2]) if self.target == "mean" else w

        outcome_loss = composite_loss(
            y, f, tau, loss_weights, f_other=f_other, tau_other=tau_other
        )
        return outcome_loss + pi_loss


# ============================================================
# Causal Ensemble
# ============================================================


class GenTE:
    """Ensemble of causal IQN models.

    Parameters
    ----------
    model_cls : {CausalIQN, CausalIQNv2}
        Causal model class.
    model_kwargs : dict, optional
        Arguments passed to the model constructor.
    n_models : int
        Number of independently initialized models.
    device : str
        Device.
    target : {"median", "mean"}
        "median" keeps the original implementation.  "mean" activates a
        quantile-specific control baseline, zero L1/MAE weight, and the mean
        over quantiles for CATE.
    """

    def __init__(self, model_cls=CausalIQNv2, model_kwargs=None, n_models=3,
                 device="auto", target="median"):
        if target not in {"mean", "median"}:
            raise ValueError("target must be 'mean' or 'median'")

        self.target = target
        self.n_models = n_models
        self.device = get_device() if device == "auto" else torch.device(device)

        # Copy so we do not mutate the user's dictionary
        model_kwargs = dict(model_kwargs) if model_kwargs else {}
        model_kwargs["target"] = target

        self.models = []
        for i in range(n_models):
            torch.manual_seed(42 + i * 137)
            self.models.append(model_cls(**model_kwargs).to(self.device))

    def fit(self, X, Y, Z, epochs=5000, lr=5e-4, weight_decay=3e-3,
            loss_weights=(0.3, 0.1, 0.6), verbose=True):
        """Train all ensemble members."""
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)

        for m_idx, model in enumerate(self.models):
            if verbose:
                print(f"  Training model {m_idx + 1}/{self.n_models}...")

            opt = optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
            sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)

            model.train()
            for epoch in range(epochs):
                opt.zero_grad()
                loss = model.loss_fn(x_t, y_t, z_t, loss_weights)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                if verbose and (epoch + 1) % 1000 == 0:
                    print(f"    Epoch {epoch + 1}/{epochs}, "
                          f"loss: {loss.item():.4f}")

    def estimate_cate(self, X, n_mc=500):
        """Estimate CATE.

        target="mean" averages the treatment effect over sampled q;
        target="median" preserves the original median summary.
        """
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        z_dum = torch.ones(X.shape[0], device=self.device)

        all_samples = []
        for model in self.models:
            model.eval()
            s = torch.zeros(X.shape[0], n_mc, device=self.device)
            with torch.no_grad():
                for i in range(n_mc):
                    tau = torch.rand(1).item()
                    _, _, te, _ = model(x_t, z_dum, tau)
                    s[:, i] = te[:, 1]
            all_samples.append(s)

        samples = torch.cat(all_samples, dim=1).cpu().numpy()

        # ONLY change for target="mean"
        if self.target == "mean":
            cate = np.mean(samples, axis=1)
        else:
            cate = np.median(samples, axis=1)

        ate_samples = samples.mean(axis=0)

        return {
            "cate": cate,
            "cate_samples": samples,
            "ci_lo": np.percentile(samples, 5, axis=1),
            "ci_hi": np.percentile(samples, 95, axis=1),
            "ate": float(cate.mean()),
            "ate_se": float(np.std(ate_samples)),
            "ate_ci": (float(np.percentile(ate_samples, 2.5)),
                       float(np.percentile(ate_samples, 97.5))),
        }

    def estimate_qte(self, X, quantiles=None):
        """Estimate quantile-indexed treatment effects.

        With target="mean" these have the direct interpretation
        Q_1(q | x) - Q_0(q | x).
        """
        if quantiles is None:
            quantiles = np.linspace(0.05, 0.95, 19)

        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        z_dum = torch.ones(X.shape[0], device=self.device)
        qte = np.zeros((X.shape[0], len(quantiles)))

        for model in self.models:
            model.eval()
            with torch.no_grad():
                for j, q in enumerate(quantiles):
                    _, _, te, _ = model(x_t, z_dum, float(q))
                    qte[:, j] += te[:, 1].cpu().numpy()

        qte /= self.n_models
        return qte

    def estimate_qte_separate(self, X, Y, Z, quantiles=None, epochs=3000,
                              hdim=128, nh=32):
        """Estimate QTE by fitting separate IQNs to treated and control."""
        if quantiles is None:
            quantiles = np.linspace(0.05, 0.95, 19)

        X1, Y1 = X[Z == 1], Y[Z == 1]
        X0, Y0 = X[Z == 0], Y[Z == 0]

        m1, xm1, xs1, ym1, ys1 = train_iqn(
            X1, Y1, epochs=epochs, hdim=hdim, nh=nh, seed=42
        )
        m0, xm0, xs0, ym0, ys0 = train_iqn(
            X0, Y0, epochs=epochs, hdim=hdim, nh=nh, seed=43
        )

        X1t = torch.tensor((X - xm1) / xs1, dtype=torch.float32)
        X0t = torch.tensor((X - xm0) / xs0, dtype=torch.float32)

        qte = np.zeros((X.shape[0], len(quantiles)))
        with torch.no_grad():
            for j, q in enumerate(quantiles):
                f1 = m1(X1t, float(q))[:, 1].numpy() * ys1 + ym1
                f0 = m0(X0t, float(q))[:, 1].numpy() * ys0 + ym0
                qte[:, j] = f1 - f0

        return qte

    def predict_propensity(self, X):
        """Predict P(Z=1|X), averaged over ensemble members."""
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        z_dum = torch.zeros(X.shape[0], device=self.device)
        pi_sum = np.zeros(X.shape[0])

        for model in self.models:
            model.eval()
            with torch.no_grad():
                _, pi_logit, _, _ = model(x_t, z_dum, 0.5)
                pi_sum += torch.sigmoid(pi_logit.view(-1)).cpu().numpy()

        return pi_sum / self.n_models