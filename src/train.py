"""
This function contains the logic that we use to train our model and improve its
internal weights. 

Author: Keon Roohparvar
Date: April 6, 2022

"""

# Python Imports
from tqdm import tqdm
import numpy as np

# Local Imports
from environment import get_env
from model_architecture import DQNAgent
from model_helper_functions import find_max_Q, find_reward

# Environment Variables
EPISODES = 500_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.15

# Reward Settings
TWO_MULT = 1
THREE_MULT = 5
FOUR_MULT = 10
FIVE_MULT = 10000
OPPONENT_MULT = 1.1
mults = [TWO_MULT, THREE_MULT, FOUR_MULT, FIVE_MULT]

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = False

# Save Model Settings
MODEL_NAME = 'BestGomoku'
MODEL_CHECKPOINT_SIZE = 100
model_num = 0


env = get_env(15)
agent = DQNAgent(model_weights='//home/kroohpar/csc487/models/BestGomoku__-99819.10max_-100079.13avg_-100880.30min__1646430535weights-1.h5')

ep_rewards = [0]
env.reset()
# Actual Training
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Flag for if there is an error in this specific episode.
    exit_episode = False

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            qs = agent.get_qs(current_state)
            action = find_max_Q(qs, env.action_space.valid_spaces)
        else:
            # Get random action
            action = np.random.choice(env.action_space.valid_spaces)

        try:
          new_state, _, done, info = env.step(action)
          reward = find_reward(env, env.player_color, mults, OPPONENT_MULT)
        except:
          exit_episode = True
        
        if exit_episode:
          break


        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    if not exit_episode:
      # Append episode reward to a list and log stats (every given number of episodes)
      ep_rewards.append(episode_reward)
      if not episode % AGGREGATE_STATS_EVERY or episode == 1:
          average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
          min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
          max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
          print(f"Average reward last {AGGREGATE_STATS_EVERY} steps: {average_reward}")
          agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

      # Save model, but only when min reward is greater or equal a set value
      if not episode % MODEL_CHECKPOINT_SIZE:
        agent.model.save_weights(f'/datasets/kroohpar/final_models/{MODEL_NAME+ str(model_num)}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__weights.h5')
        model_num += 1
        
      # Decay epsilon
      if epsilon > MIN_EPSILON:
          epsilon *= EPSILON_DECAY
          epsilon = max(MIN_EPSILON, epsilon)