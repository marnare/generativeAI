import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Stefan-Boltzmann constant in W/m²K⁴
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8

class TreatmentEffectNet(nn.Module):
    """
    Neural network for estimating heterogeneous treatment effects using quantile regression.
    
    The network estimates:
    1. Propensity scores (probability of treatment assignment)
    2. Baseline outcomes (expected outcome without treatment)
    3. Treatment effects at different quantiles
    
    Args:
        x_dim (int): Dimension of covariates
        hidden_dim (int, optional): Size of hidden layers. Defaults to 256.
    
    Input variables:
        x: Covariates/features for each unit
        z: Binary treatment assignment (0 = control, 1 = treated)
        tau: Quantile level (0 to 1) for treatment effect estimation
        y: Observed outcomes
    """
       
    def __init__(self, x_dim, hidden_dim=256):
        super().__init__()
        pisz = 8
        self.nh = 32  # number of basis functions
        
        # Propensity network (pi)
        self.pi = nn.Sequential(
            nn.Linear(x_dim, 16),
            nn.ReLU(),
            nn.Linear(16, pisz)
        )
        self.pi1 = nn.Sequential(
            nn.Linear(pisz, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Baseline network (mu)
        self.mu = nn.Sequential(
            nn.Linear(x_dim + pisz, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )

        self.mu1 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Treatment effect network (te)
        self.te = nn.Sequential(
            nn.Linear(x_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        self.te1 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.tau = nn.Sequential(
            nn.Linear(self.nh, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, z, tau):

        """
        Forward pass of the network.
        
        Args:
            x: Covariates tensor
            z: Treatment assignment tensor (binary)
            tau: Quantile level (between 0 and 1)
            
        Returns:
            y: Predicted outcomes
            pi1: Propensity scores
            te: Treatment effects
            mu: Baseline predictions
        """
        # Ensure z is a tensor with shape (batch_size, 1)
        device = x.device
        dtype = x.dtype
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z, dtype=dtype, device=device)
        z = z.to(device=device, dtype=dtype)
        if z.dim() == 0:
            z = z.expand(x.size(0), 1)
        else:
            z = z.reshape(-1, 1)

        # Basis expansion for treatment effects
        tau = torch.cos(torch.arange(start=0, end=self.nh) * torch.pi * tau) ## simulates cos(k, pi, q) for each k in [0, 32]. 
        tau = self.tau(tau) # ReLU is applied to each element of the vector
        
        # Propensity score
        pi = self.pi(x) # the 8-dimensional intermediate output (embeddings)
        pi1 = self.pi1(pi) # the propensity score (probability of treatment assignment), i.e. logit(pi)
        
        
        # Baseline
        mu = self.mu1(self.mu(torch.cat((x, pi), 1)))
        
    
        # Sample from normal distribution with mean = 0
        # and standard deviation = 0.1 * Stefan-Boltzmann radiation (10% variance)
        # Calculate Stefan-Boltzmann radiation
        # Assuming temperature is encoded in the first feature of x
        temperature = x[:, 0]  # Extract temperature from first feature
        stefan_boltzmann_radiation = STEFAN_BOLTZMANN_CONSTANT * torch.pow(temperature, 4)

        # Treatment effect
        te = self.te1(tau * self.te(x))  # current treatment effect

        # Rescale variance to match luminosity scale (L/Lo)
        # c = 1e-15  # calibration constant to match empirical scale
        # noise = torch.randn_like(y, device=x.device).view(-1, 1)

        std = 1.0
        noise = std * torch.randn(x.size(0), 1, device=x.device, dtype=x.dtype)

        
        # Final output (z already shape (-1, 1) from above)
        y = mu + te * z + noise
        return y, pi1, te, mu

    

    def loss_fn(self, x, y, z, w):
        """
        Computes the loss function combining multiple objectives:
        
        1. Propensity score estimation using binary cross-entropy
        2. Baseline outcome prediction using absolute error
        3. Quantile regression loss for treatment effects
        
        Args:
            x: Covariates tensor
            y: Observed outcomes tensor
            z: Treatment assignment tensor
            w: Loss weights for different components [baseline_weight, quantile_weight, treatment_weight]
            
        Returns:
            Combined loss value
        """
        zlossfn = nn.BCELoss()
        tau = torch.rand(1).item() ## samples a random quantile between 0 and 1
        tauind = tau < 0.5
        
        f, pi, _, _ = self(x, z, tau)
        piloss = zlossfn(torch.sigmoid(pi.view(-1)), z)
        
        e = y.view(-1, 1) - f  # error term (single output: f = mu + te*z)
        e_ = e.view(-1)
        loss = w[0] * torch.mean(torch.abs(e_))  # baseline / prediction error
        loss += w[1] * torch.abs(torch.tensor(tau-0.5)) * (
            tauind * torch.mean(torch.relu(-e_)) + (1-tauind) * torch.mean(torch.relu(e_))
        )
        loss += w[2] * torch.mean(torch.maximum(tau*e_, (tau-1)*e_))  # pinball loss for treatment effect
        loss += piloss
        return loss




class GenTE:
    """
    Generative treatment effect estimator. Wraps TreatmentEffectNet with fit/effect API.
    
    Args:
        epochs (int): Number of training epochs. Default 100.
        batch_size (int): Minibatch size. Default 32.
        lr (float): Adam learning rate. Default 0.001.
        loss_weights (list): [baseline_weight, quantile_weight, treatment_weight]. Default [1.0, 0.1, 0.1].
        hidden_dim (int): Hidden layer size for TreatmentEffectNet. Default 256.
        random_state (int, optional): Random seed for reproducibility.
    """
    def __init__(self, epochs=100, batch_size=32, lr=0.001, loss_weights=None,
                 hidden_dim=256, random_state=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_weights = loss_weights if loss_weights is not None else [1.0, 0.1, 0.1]
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.model_ = None
        self.x_dim_ = None

    def fit(self, X, D, Y):
        """
        Fit GenTE on training data.

        Args:
            X: Covariates (n_samples, n_features).
            D: Treatment assignment (n_samples,), binary 0/1.
            Y: Observed outcomes (n_samples,).

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        D = np.asarray(D, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if self.random_state is not None:
            set_all_seeds(self.random_state)

        self.x_dim_ = X.shape[1]
        self.model_ = TreatmentEffectNet(x_dim=self.x_dim_, hidden_dim=self.hidden_dim)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        n = len(X)

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            for i in range(0, n, self.batch_size):
                batch_idx = idx[i:i + self.batch_size]
                batch_X = torch.FloatTensor(X[batch_idx])
                batch_D = torch.FloatTensor(D[batch_idx])
                batch_Y = torch.FloatTensor(Y[batch_idx])
                optimizer.zero_grad()
                self.model_.loss_fn(batch_X, batch_Y, batch_D, self.loss_weights).backward()
                optimizer.step()
        return self

    def effect(self, X):
        """
        Predict CATE for test units.

        Args:
            X: Covariates (n_samples, n_features).

        Returns:
            Predicted CATE (n_samples,).
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        tau_pred = []
        with torch.no_grad():
            for i in range(len(X)):
                _, _, te, _ = self.model_(
                    torch.FloatTensor(X[i:i + 1]),
                    torch.ones(1),
                    0.5
                )
                tau_pred.append(te.numpy()[0, 0])
        return np.array(tau_pred)


# Set all seeds
def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed) 


