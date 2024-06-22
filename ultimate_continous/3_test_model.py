"""Test the trained policy (for REINFORCE).

Author: Elie KADOCHE.
"""

import torch

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.envs.cartpole_v1 import CartpoleEnvV1
from src.models.actor_v0 import ActorModelV0
from src.models.actor_v1 import ActorModelV1

from torch.distributions import Normal

STD = 0.1

if __name__ == "__main__":
    # Create environment and policy
    env = CartpoleEnvV0()
    policy = ActorModelV0()
    actor_path = "./saved_models/actor_0.pt"

    # ------------------------------------------
    # ---> TODO: UNCOMMENT FOR SECTION 4 ONLY
    # env = CartpoleEnvV1()
    # policy = ActorModelV1()
    # actor_path = "./saved_models/actor_1.pt"
    # ------------------------------------------

    # Testing mode
    policy.eval()
    print(policy)

    # Load the trained policy
    policy = torch.load(actor_path)

    # Reset it
    total_reward = 0.0
    state, _ = env.reset(seed=None)

    # While the episode is not finished
    terminated = False
    while not terminated:

        # Convert state to tensor, take mean, define gaussian, sample from the gaussian, clamp, compute log probability of the clamped action, same like reinforce
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        mean = policy(state_tensor)
        normal_dist = Normal(mean, STD)
        action = normal_dist.sample().item()
        action_clamped = torch.clamp(torch.tensor(action), -1.0, 1.0)




        # One step forward
        state, reward, terminated, _, _ = env.step(action_clamped)

        # Render (or not) the environment
        total_reward += reward
        env.render()

    # Print reward
    print("total_reward = {}".format(total_reward))
