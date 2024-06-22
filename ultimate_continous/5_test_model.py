"""Test the trained policy (for actor-critic).

Author: Elie KADOCHE.
"""

import torch

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.models.actor_v1 import ActorModelV1
from torch.distributions import Normal

STD = 0.1 # Standard deviation is a hyperparameter set to 0.1

if __name__ == "__main__":
    # Create policy
    policy = ActorModelV1()
    policy.eval()
    print(policy)

    # Load the trained policy
    policy = torch.load("./saved_models/actor_2.pt")

    # Create environment
    env = CartpoleEnvV0()

    # Reset it
    total_reward = 0.0
    state, _ = env.reset(seed=None)

    # While the episode is not finished
    terminated = False
    while not terminated:

        state_tensor = torch.from_numpy(state).float().unsqueeze(0) # convert state to tensr
        mean = policy(state_tensor) # access mean from the file
        normal_dist = Normal(mean, STD) # define a gaussian distribution
        action = normal_dist.sample().item() # sample an action from the normal distribution
        action_clamped = torch.clamp(torch.tensor(action), -1.0, 1.0) # clamp the action
        

        # One step forward
        state, reward, terminated, _, _ = env.step(action)

        # Render (or not) the environment
        total_reward += reward
        env.render()

    # Print reward
    print("total_reward = {}".format(total_reward))
