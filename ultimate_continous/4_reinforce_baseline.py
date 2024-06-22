"""REINFORCE (with baseline) algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim
from torch.distributions import Normal
import torch.nn.functional as F

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.models.actor_v0 import ActorModelV0
from src.models.critic import CriticModel

HORIZON = 500
STD = 0.1 # standard deviation
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0001
STOP_THRESOLD = 10 # stop thresold for termination i.e. 10 conescutive 500 reward implies termination


if __name__ == "__main__":
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    critic = CriticModel()
    actor_path = "./saved_models/actor_2.pt"
    critic_path = "./saved_models/critic_2.pt"
    actor.train()
    critic.train()

    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    consecutive_reward_count = 0
    training_iteration = 0

    while True:
        state, _ = env.reset()
        saved_log_probs = []
        saved_rewards = []
        saved_values = []
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)


        for t in range(HORIZON + 1):
            mean = actor(state_tensor) # access mean from actor neural net
            normal_dist = Normal(mean, STD) # define a distribution
            action = normal_dist.sample() # sample from the gaussian distribution
            action_clamped = torch.clamp(action, -1.0, 1.0) # clamp the action so its in the range from 1 to -1
            log_prob = normal_dist.log_prob(action_clamped) # find the log probability of clamped action from the gaussian
            

            next_state, reward, terminated, _, _ = env.step(action_clamped.item())
            saved_log_probs.append(log_prob)
            saved_rewards.append(reward)
            saved_values.append(critic(state))

            if terminated:
                break
            state = next_state

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

        normalized_delta = (delta - delta.mean())/(delta.std() + 1e-7) # standardizing delta


        # For each time step
        actor_loss = list()
        for p, g in zip(saved_log_probs, normalized_delta):

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
        # We stop the training when reward is 500 for 10 consecutive iteration

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
