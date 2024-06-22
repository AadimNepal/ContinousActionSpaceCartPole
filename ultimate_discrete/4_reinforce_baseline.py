"""REINFORCE (with baseline) algorithm.

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
    actor_path = "./saved_models/actor_2.pt"
    critic_path = "./saved_models/critic_2.pt"

    # Training mode
    actor.train()
    critic.train()
    print(actor)
    print(critic)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # Create optimizer with the critic parameters
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # ---> TODO: based on the REINFORCE script, create the REINFORCE with
    # baseline script

    consecutive_reward_count = 0
    training_iteration = 0

    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri
        saved_probabilities = list()
        saved_rewards = list()
        saved_values = list()

        # Prevent infinite loop
        for t in range(HORIZON + 1):

            # Use the policy to generate the probabilities of each action
            probabilities = actor(state)

            # Create a categorical distribution over the list of probabilities
            # of actions and sample an action from it
            distribution = Categorical(probabilities)
            action = distribution.sample()

            # Take the action
            state, reward, terminated, _, _ = env.step(action.item())

            # Save the probability of the chosen action and the reward
            saved_probabilities.append(probabilities[0][action])
            saved_rewards.append(reward)
            saved_values.append(critic(state))

            # End episode
            if terminated:
                break

        # Compute discounted sum of rewards
        # ------------------------------------------
        # Current discounted reward
        discounted_reward = 0
        discounted_rewards = []

# Process rewards from last to first using enumerate on the reversed list
        for i, r in enumerate(reversed(saved_rewards)):
            # Compute the discounted reward at each time step
            # Keep a cumulative sum of the discounted rewards
            discounted_reward = r + DISCOUNT_FACTOR * discounted_reward

            # Insert the computed discounted reward at the beginning of the list
            discounted_rewards.insert(0, discounted_reward)

        # Eventually normalize for stability purposes
        discounted_rewards = torch.tensor(discounted_rewards)
        mean, std = discounted_rewards.mean(), discounted_rewards.std()
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)

        # Update policy parameters
        # ------------------------------------------

        # Now we will compute delta
        state_values = torch.tensor(saved_values, requires_grad=True)
        discounted_rewards = torch.tensor(discounted_rewards)

        delta = discounted_rewards - state_values

        normalized_delta = (delta - delta.mean())/(delta.std() + 1e-7)


        # For each time step
        actor_loss = list()
        for p, g in zip(saved_probabilities, normalized_delta):

            # ---> TODO: compute policy loss
            # The policy graident theorem defines the loss function as follows:
            # Since we have the baselines now, we multiply as follows

            time_step_actor_loss = -torch.log(p) * g

            # Save it
            actor_loss.append(time_step_actor_loss)
        

        # Sum all the time step losses
        actor_loss = torch.cat(actor_loss).sum()
        critic_loss = F.mse_loss(state_values, discounted_rewards)

        # Reset gradients to 0.0
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        actor_loss.backward()
        critic_loss.backward()

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()
        critic_optimizer.step()

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?
        # When 500 reward is reached consecutovely for 10 iterations

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
            # When 500 reward is reached for 10 consecutive iterations

