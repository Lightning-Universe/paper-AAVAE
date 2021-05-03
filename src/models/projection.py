import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHeadAE(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super(ProjectionHeadAE, self).__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        return self.projection_head(x)


class ProjectionHeadVAE(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super(ProjectionHeadVAE, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, output_dim, bias=False)
        self.logvar = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.first_layer(x)
        return self.mu(x), self.logvar(x)
