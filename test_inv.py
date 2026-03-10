import torch
import torch.nn as nn

class CouplingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.mlp(x)

class AffineCouplingLayer(nn.Module):
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
    def __init__(self, n_layers=8, hidden=128):
        super().__init__()
        # stack K coupling layers with alternating parity
        self.layers = nn.ModuleList([
            AffineCouplingLayer(hidden, parity=i % 2) 
            for i in range(n_layers)
        ])
        
    def forward(self, z):
        """Map base distribution z to data x (sampling)."""
        log_det_tot = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            z, log_det = layer(z, inverse=False)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, x):
        """Map data x to base distribution z."""
        log_det_tot = torch.zeros(x.shape[0], device=x.device)
        # reverse order for the inverse map
        for layer in reversed(self.layers):
            x, log_det = layer(x, inverse=True)
            log_det_tot += log_det
        return x, log_det_tot

import numpy as np
import pandas as pd

if __name__ == "__main__":
    torch.manual_seed(42)
    model = NormFlow(n_layers=8, hidden=64)
    train_df = pd.read_csv("data/moons_train.csv")
    x = torch.tensor(train_df[['x1', 'x2']].values).float()
    
    # Invertibility max abs error
    z, _ = model.inverse(x)
    x_hat, _ = model.forward(z)
    
    max_err = torch.max(torch.abs(x - x_hat)).item()
    print(f"Max abs error: {max_err}")

    x0 = x[0] # first example
    epsilon = 1e-4
    J_num = torch.zeros(2, 2)
    for i in range(2):
        x_plus = x0.clone()
        x_plus[i] += epsilon
        x_minus = x0.clone()
        x_minus[i] -= epsilon
        
        z_plus, _ = model.inverse(x_plus.unsqueeze(0))
        z_minus, _ = model.inverse(x_minus.unsqueeze(0))
        
        J_num[:, i] = (z_plus.squeeze(0) - z_minus.squeeze(0)) / (2 * epsilon)

    log_det_num = torch.log(torch.abs(torch.det(J_num))).item()
    _, log_det_analytic_tensor = model.inverse(x0.unsqueeze(0))
    log_det_analytic = log_det_analytic_tensor.item()
    abs_error_logdet = abs(log_det_num - log_det_analytic)
    print(f"Log-det abstract error: {abs_error_logdet}")

