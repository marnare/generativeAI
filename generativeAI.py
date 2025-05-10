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
            nn.Linear(32, 2)
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

        # Basis expansion for treatment effects
        tau = torch.cos(torch.arange(start=0, end=self.nh) * torch.pi * tau) ## simulates cos(k, pi, q) for each k in [0, 32]. 
        tau = self.tau(tau) # ReLU is applied to each element of the vector
        
        # Propensity score
        pi = self.pi(x)
        pi1 = self.pi1(pi)
        
        # Baseline
        mu = self.mu1(self.mu(torch.cat((x, pi), 1)))
        
    
        # Sample from normal distribution with mean = stefan_boltzmann_radiation
        # and standard deviation = 0.1 * mean (10% variance)
        # Calculate Stefan-Boltzmann radiation
        # Assuming temperature is encoded in the first feature of x
        temperature = x[:, 0]  # Extract temperature from first feature
        stefan_boltzmann_radiation = STEFAN_BOLTZMANN_CONSTANT * torch.pow(temperature, 4)

        # Treatment effect
        te = self.te1(tau * self.te(x))  # Your current treatment effect

        noise = torch.normal(
            mean=stefan_boltzmann_radiation.view(-1, 1),
            std=0.1 * stefan_boltzmann_radiation.view(-1, 1)
        )

        
        # Final output
        y = mu + te * z.view(-1, 1) + noise
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
        tau = torch.rand(1).item()
        tauind = tau < 0.5
        
        f, pi, _, _ = self(x, z, tau)
        piloss = zlossfn(torch.sigmoid(pi.view(-1)), z)
        
        e = y.view(-1, 1) - f # error term for minimizing the weighted loss. 
        loss = w[0] * torch.mean(torch.abs(e[:, 0])) # loss for the baseline. 
        loss += w[1] * torch.abs(torch.tensor(tau-0.5)) * (
            tauind * torch.mean(torch.relu(-e[:, 1])) +  # quantile adjustment for crossing the quantile. 
            (1-tauind) * torch.mean(torch.relu(e[:, 1]))
        )
        loss += w[2] * torch.mean(torch.maximum(tau*e[:, 1], (tau-1)*e[:, 1])) # loss for the treatment effect. 
        loss += piloss # loss for the propensity score. 
        return loss




# Set all seeds
def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)