"""REINFORCE algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.envs.cartpole_v1 import CartpoleEnvV1
from src.models.actor_v0 import ActorModelV0
from src.models.actor_v1 import ActorModelV1

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.001

STOP_THRESOLD = 10 #thresold to stop for convergence, when 500 reward is reached consecutively for 10 iterations

if __name__ == "__main__":
    # Create environment and policy
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    actor_path = "./saved_models/actor_0.pt"

    # ------------------------------------------
    # ---> TODO: UNCOMMENT FOR SECTION 4 ONLY
    env = CartpoleEnvV1()
    actor = ActorModelV1()
    actor_path = "./saved_models/actor_1.pt"
    # ------------------------------------------

    # Training mode
    actor.train()
    print(actor)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # ---> TODO: when do we stop the training?
    # We stop it when the reward converges, more details below
    # Run infinitely many episodes
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


        # For each time step
        actor_loss = list()
        for p, g in zip(saved_probabilities, discounted_rewards):

            # ---> TODO: compute policy loss
            # The policy graident theorem defines the loss function as follows:
            # We want max which is equivalent to finding min of negative, so we multiply by -1

            time_step_actor_loss = -torch.log(p) * g

            # Save it
            actor_loss.append(time_step_actor_loss)
        

        # Sum all the time step losses
        actor_loss = torch.cat(actor_loss).sum()

        # Reset gradients to 0.0
        actor_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        actor_loss.backward()

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?
        # Stop when the reward reaches 500 consecutively for 10 iterations

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

            # Print results
            print("iteration {} - last reward: {:.2f}".format(
                training_iteration, episode_total_reward))

            # ---> TODO: when do we stop the training?
            # Stop when the reward reaches 500 consecutively for 10 iterations

