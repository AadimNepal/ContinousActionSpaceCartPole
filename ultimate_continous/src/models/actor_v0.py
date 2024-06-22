"""Actor (policy) deep neural network.

Author: Elie KADOCHE.
"""

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class ActorModelV0(nn.Module):
    """Deep neural network."""

    # By default, use CPU
    DEVICE = torch.device("cpu")

    def __init__(self):
        """Initialize model."""
        super(ActorModelV0, self).__init__()
        # ---> TODO: change input and output sizes depending on the environment
        input_size = 4
        action_dimension = 1 # the ouput dimension is just 1 in this case

        # Build layer objects
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dimension)
        self.std_layer = nn.Linear(128, action_dimension) # ignore this, I tried with this but didnt work, I didnt include it below

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
        mean = torch.tanh(self.mean_layer(x))
        # use tan hyperbolic so that the ouput is in the range from -1 to 1

        return mean
