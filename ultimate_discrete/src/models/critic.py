"""Critic (value) deep neural network.

Author: Elie KADOCHE.
"""

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# ---> TODO: based on the actor network, build a critic network

class CriticModel(nn.Module):
    """Deep neural network."""

    # By default, use CPU
    DEVICE = torch.device("cpu")

    def __init__(self):
        """Initialize model."""
        super(CriticModel, self).__init__()
        # ---> TODO: change input and output sizes depending on the environment
        input_size = 4

        # Build layer objects
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 128)
        self.final_value = nn.Linear(128, 1)

    def _preprocessor(self, state):
        """Preprocessor function.

        Args:
            state (numpy.array): environment state.

        Returns:
            x (torch.tensor): preprocessed state.
        """
        # Add batch dimension
        x = np.expand_dims(state, 0)

        # Transform to torch.tensor
        x = torch.from_numpy(x).float().to(self.DEVICE)

        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x (numpy.array): environment state.

        Returns:
            actions_prob (torch.tensor): list with the probability of each
                action over the action space.
        """
        # Preprocessor
        x = self._preprocessor(x)

        # Input layer
        x = F.relu(self.fc0(x))

        # Middle layers
        x = F.relu(self.fc1(x))

        # Policy
        value = self.final_value(x)

        return value
