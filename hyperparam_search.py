import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import itertools

# MLP for flow: Linear(1D -> H) -> ReLU -> Linear(H -> 2D)
class CouplingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.mlp(x)

class CouplingLayer(nn.Module):
    def __init__(self, hidden_dim, parity):
        super().__init__()
        self.parity = parity 
        # input: h (1D), output: s(h) and t(h) (2D)
        self.mlp = CouplingMLP(1, hidden_dim, 2)

    def forward(self, h, inverse=False):
        # split vector based on parity
        h1 = h[:, self.parity:self.parity+1]
        h2 = h[:, 1-self.parity:1-self.parity+1]

        # get s and t from the MLP
        stats = self.mlp(h1)
        s, t = stats.chunk(2, dim=1)
        
        # bound scale output
        s = torch.tanh(s)

        if not inverse:
            # forward z -> x
            new_h2 = h2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)
        else:
            # inverse x -> z
            new_h2 = (h2 - t) * torch.exp(-s)
            log_det = -torch.sum(s, dim=1)

        # reassemble based on parity
        if self.parity == 0:
            return torch.cat([h1, new_h2], dim=1), log_det
        else:
            return torch.cat([new_h2, h1], dim=1), log_det

class NormFlow(nn.Module):
    def __init__(self, n_layers=8, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(hidden_dim, i % 2) # alternate pparity 
            for i in range(n_layers)])

    # f[z]
    def forward(self, z):
        """Map base distribution to data (sampling)."""
        log_det_tot = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            z, log_det = layer(z, inverse=False)
            log_det_tot += log_det
        return z, log_det_tot

    # f^{-1}[x]
    def inverse(self, x):
        """Map data to base distribution (log-likelihood evaluation)."""
        log_det_tot = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            x, log_det = layer(x, inverse=True)
            log_det_tot += log_det
        return x, log_det_tot

def log_prob(model, x):
    z, log_det_J = model.inverse(x)
    # Base distribution: N(0, I)
    log_p_z = -0.5 * (np.log(2 * np.pi) + z**2).sum(dim=1)
    return log_p_z + log_det_J

def train_and_eval(n_layers, hidden_dim, train_tensor, val_tensor, lr=1e-3, steps=1000):
    torch.manual_seed(42) # Set seed for reproducibility
    model = NormFlow(n_layers=n_layers, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        loss = -log_prob(model, train_tensor).mean()
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        val_nll = -log_prob(model, val_tensor).mean().item()
        
    return val_nll

def main():
    print("Loading data...")
    # Load data
    train_df = pd.read_csv("data/moons_train.csv")
    val_df = pd.read_csv("data/moons_val.csv")
    
    # Extract features (x1, x2) as float32 tensors
    train_tensor = torch.tensor(train_df[['x1', 'x2']].values, dtype=torch.float32)
    val_tensor = torch.tensor(val_df[['x1', 'x2']].values, dtype=torch.float32)

    # Define hyperparameter grid (instructions limit to n_layers=8, hidden=128)
    layers_grid = [4, 6, 8]
    hidden_grid = [32, 64, 128]
    
    results = []
    
    print("Starting hyperparameter tuning...")
    print(f"{'n_layers':<10} | {'hidden':<8} | {'val_nll':<10}")
    print("-" * 35)
    
    for n_layers, hidden_dim in itertools.product(layers_grid, hidden_grid):
        # We'll use 2000 steps since it's a small dataset and model
        val_nll = train_and_eval(n_layers, hidden_dim, train_tensor, val_tensor, steps=2000)
        results.append({
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'val_nll': val_nll
        })
        print(f"{n_layers:<10} | {hidden_dim:<8} | {val_nll:.4f}")
        
    # Find best config
    best_config = min(results, key=lambda x: x['val_nll'])
    print("\nBest Configuration:")
    print(f"n_layers: {best_config['n_layers']}, hidden_dim: {best_config['hidden_dim']}")
    print(f"Validation NLL: {best_config['val_nll']:.4f}")

if __name__ == "__main__":
    main()
