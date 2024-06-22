"""Actor-critic algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.models.actor_v0 import ActorModelV0
from src.models.critic import CriticModel

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.001

STOP_THRESOLD = 10

if __name__ == "__main__":
    # Create environment, policy and critic
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    critic = CriticModel()
    actor_path = "./saved_models/actor_3.pt"
    critic_path = "./saved_models/critic_3.pt"

    # Training mode
    actor.train()
    critic.train()
    print(actor)
    print(critic)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # Create optimizer with the critic parameters
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # ---> TODO: based on the REINFORCE script, create the actor-critic script

    # ---> TODO: when do we stop the training?
    # Run infinitely many episodes
    consecutive_reward_count = 0
    training_iteration = 0
    running_score = 0

    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri

        saved_rewards = list()
        discount_tracker = 1

        # Prevent infinite loop
        for t in range(HORIZON + 1):

            # Use the policy to generate the probabilities of each action
            probabilities = actor(state)[0]

            # Create a categorical distribution over the list of probabilities
            # of actions and sample an action from it
            distribution = Categorical(probabilities)
            action = distribution.sample()
            state_value = critic(state)

            # Take the action
            state, reward, terminated, _, _ = env.step(action.item())
            running_score = running_score + reward
            saved_rewards.append(reward)

            # Save the probability of the chosen action and the rewar

            # End episode
            if terminated:
                new_value_state = torch.tensor([0]).float().unsqueeze(0)
            else:
                new_value_state = critic(state)

            delta = reward + DISCOUNT_FACTOR * new_value_state - state_value
            value_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_value_state, state_value)
            policy_loss = -distribution.log_prob(action) * delta.detach() * discount_tracker
        

            # Reset gradients to 0.0
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            # Compute the gradients of the loss (backpropagation)
            policy_loss.backward()
            value_loss.backward()

            # Update the policy parameters (gradient ascent)
            actor_optimizer.step()
            critic_optimizer.step()

            if terminated:
                break

            discount_tracker = discount_tracker * DISCOUNT_FACTOR

            # Logging
            # ------------------------------------------

            # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?
        # We stop when I reach 500 reward for 10 iterations consecutively
        if episode_total_reward == HORIZON:
            consecutive_reward_count = consecutive_reward_count + 1
        else:
            consecutive_reward_count = 0

        if consecutive_reward_count == STOP_THRESOLD:
            break


        # Log results
        log_frequency = 5
        training_iteration += 1
        if training_iteration % log_frequency == 0:

            # Save neural network
            torch.save(actor, actor_path)
            torch.save(critic, critic_path)

            # Print results
            print("iteration {} - last reward: {:.2f}".format(
                training_iteration, episode_total_reward))

            # ---> TODO: when do we stop the training?

        


