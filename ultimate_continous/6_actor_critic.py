import torch
from torch import optim
from torch.distributions import Normal
import torch.nn.functional as F

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.models.actor_v0 import ActorModelV0
from src.models.critic import CriticModel
import numpy as np

HORIZON = 500 # specifies the lenght of each episode
STD = 0.1 # hyperparameter, standard deviation set to 0.1
DISCOUNT_FACTOR = 0.99 # gamma
LEARNING_RATE = 0.0001 # alpha
STOP_THRESHOLD = 10 # thresold to stop the convergence, when there are 10 consecutive 500 rewards

if __name__ == "__main__":
    # create object for env, actor, critic
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    critic = CriticModel()
    actor_path = "./saved_models/actor_3.pt"
    critic_path = "./saved_models/critic_3.pt"
    actor.train()
    critic.train()


    # Define an Adam optimizer for actor and critic
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE) 
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    consecutive_reward_count = 0
    training_iteration = 0

    while consecutive_reward_count < STOP_THRESHOLD: #this is basically specifying when to terminate, converges when consecutive reward count is equal to 10
        reset_output = env.reset()
        state = reset_output if isinstance(reset_output, np.ndarray) else reset_output[0]  # Adjust based on actual structure
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        episode_rewards = []

        for t in range(HORIZON):
            #extract mean, define distribution, sample and compute liklyhood of tha sample, clamp, similar to the comments in reinforce
            mean = actor(state_tensor)
            dist = Normal(mean, STD) #define a normal distribution with mean and STD = 0.1
            action = dist.sample()
            action_clamped = torch.clamp(action, -1.0, 1.0) # clamp the action
            log_prob = dist.log_prob(action_clamped) # compute the log probability of clamped action from the defined gaussian
            next_state, reward, done, _, _ = env.step(action_clamped.item())
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

            # Reward collection for the episode
            episode_rewards.append(reward)

            # Critic evaluations
            state_value = critic(state_tensor)
            next_state_value = critic(next_state_tensor) if not done else torch.tensor([0.0])

            # Calculate deltas
            delta = reward + DISCOUNT_FACTOR * next_state_value - state_value

            # Policy and value loss
            policy_loss = -log_prob * delta.detach()
            value_loss = F.mse_loss(state_value, reward + DISCOUNT_FACTOR * next_state_value)

            # Optimize
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

            if done:
                break

            state_tensor = next_state_tensor

        # Tracking and logging
        total_reward = sum(episode_rewards)
        training_iteration += 1
        # If the rewardis greater than or equal to 500 consecitvely for 10 iterations, then convergennce has been attained. Increment a  counter, else set it to 0, temrinates when counter is greater than or equal to 10
        if total_reward >= HORIZON:
            consecutive_reward_count += 1
        else:
            consecutive_reward_count = 0

        if training_iteration % 5 == 0:
            print(f"Iteration {training_iteration} - last reward: {total_reward:.2f}")
            torch.save(actor, actor_path) # save the weights in the specified paths
            torch.save(critic, critic_path)

    print("Training completed")

