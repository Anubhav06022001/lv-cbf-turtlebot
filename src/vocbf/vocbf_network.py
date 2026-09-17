import torch
import torch.nn as nn

class BarrierNet(nn.Module):
    def __init__(self, state_dim=5, hidden_dim=128):
        super(BarrierNet, self).__init__()

        # Pure State MLP: Takes 5D state and directly outputs 1D safety value
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        # The input 'state' is the normalized 5D vector
        B_val = self.mlp(state)
        return B_val