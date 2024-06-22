"""Test a random policy.

Author: Elie KADOCHE.
"""

import torch
from torch.distributions import Normal
from src.envs.cartpole_v0 import CartpoleEnvV0
from src.envs.cartpole_v1 import CartpoleEnvV1
from src.models.actor_v0 import ActorModelV0
from src.models.actor_v1 import ActorModelV1
import numpy as np

if __name__ == "__main__":
    # Create environment and policy
    env = CartpoleEnvV0()
    policy = ActorModelV0()

    # Testing mode
    policy.eval()
    print(policy)

    # Reset it
    total_reward = 0.0
    state, _ = env.reset(seed=None)
    STD = 0.1

    # While the episode is not finished
    terminated = False
    while not terminated:

        # Use the policy to generate the probabilities of each action
        state_tensor = torch.from_numpy(state).float().unsqueeze(0) # convert state to tensor
        mean = policy(state_tensor) # get mean value from actor neural net
        normal_dist = Normal(mean, STD) # define a gaussian distribution of mean and STD = 0.1
        action = normal_dist.sample().item() # sample a value from the gaussian
        
        # clipping the action values in the range -1 to 1
        if action > 1.0:
            action = 1.0
        elif action < -1.0:
            action = -1.0

        # One step forward
        state, reward, terminated, _, _ = env.step(action)

        # Render (or not) the environment
        total_reward += reward
        env.render()

    # Print reward
    print("total_reward = {}".format(total_reward))
