"""REINFORCE algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Normal

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.envs.cartpole_v1 import CartpoleEnvV1
from src.models.actor_v0 import ActorModelV0
from src.models.actor_v1 import ActorModelV1

STD = 0.1 # hyperparamter for standard deviation 

# Maximum environment length
HORIZON = 400

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.0001

STOP_THRESOLD = 10

if __name__ == "__main__":
    # Create environment and policy
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    actor_path = "./saved_models/actor_0.pt"


    # Training mode
    actor.train()
    print(actor)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # ---> TODO: when do we stop the training?
    # Run infinitely many episodes
    consecutive_reward_count = 0
    training_iteration = 0
    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri
        saved_probabilities = list()
        saved_rewards = list()
    


        # Prevent infinite loop
        for t in range(HORIZON + 1):

           
            mean = actor(state_tensor) # take mean from actor neural net
            normal_dist = Normal(mean, STD) # define a distribution with mean and STD = 0.1
            action = normal_dist.sample() # sample from the distribution
            action_clamped = torch.clamp(action, -1.0, 1.0) # clamp in the specified range
            log_probability = normal_dist.log_prob(action_clamped) # compute the log probability of clamped action from the gaussian distribution
            
    
            
            # Take the action
            state, reward, terminated, _, _ = env.step(action_clamped.item())

            # Save the probability of the chosen action and the reward
            saved_probabilities.append(log_probability)
            saved_rewards.append(reward)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)

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

        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?


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
        # Stop the training when reward is greater than 300 consecutively for 10 times
        if episode_total_reward >= 300:
            consecutive_reward_count = consecutive_reward_count + 1
        else:
            consecutive_reward_count = 0

        if consecutive_reward_count >= 10:
            break

