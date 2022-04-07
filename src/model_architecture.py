"""
This script contains the model architecture's logic, and some helper functinoality to help 
interface the neural network with the gym environment.

Author: Keon Roohparvar
Date: April 6, 2022

"""

# Python Imports
import tensorflow as tf
from keras import Sequential
from keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import random


# Local Imports
from model_helper_functions import *
from modified_tensorboard import ModifiedTensorBoard



DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MIN_REWARD = -5000  # For model save
MEMORY_FRACTION = 0.20
MODEL_NAME = 'BestGomoku'
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5 
BOARD_SIZE = 15

class DQNAgent:
  def __init__(self, env, model_location=None, model_weights=None):
    #Save environment
    self.env = env


    # Initialize random Q Model
    if model_location == None:
      self.model = self.create_model()

    # Using pretrained model
    else:
      self.model = tf.keras.models.load_model(model_location)
    
    # Load weights in if weights file is given
    if model_weights != None:
      self.model.load_weights(model_weights)

    # Our Target Q Model
    self.target_model = self.create_model()
    self.target_model.set_weights(self.model.get_weights())


    # Our Replay Memory which stores previous actions
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    # Custom tensorboard object
    self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    # Used to count when to update target network with main network's weights
    self.target_update_counter = 0
  
  def create_model(self):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(self.env.board_size, self.env.board_size)))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))

    model.add(Dense(self.env.board_size**2, activation='linear'))  
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

  # Adds step's data to a memory replay array
  # (observation space, action, reward, new observation space, done)
  def update_replay_memory(self, transition):
    self.replay_memory.append(transition)
  
  # Trains main network every step during episode
  def train(self, terminal_state, step):
    # Start training only if certain number of samples is already saved
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    
    # Get a minibatch of random samples from memory replay table
    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

    # Get current states from minibatch, then query NN model for Q values
    current_states = np.array([transition[0] for transition in minibatch])
    current_qs_list = self.model.predict(current_states)

    # Get future states from minibatch, then query NN model for Q values
    # When using target network, query it, otherwise main network should be queried
    new_current_states = np.array([transition[3] for transition in minibatch])
    future_qs_list = self.target_model.predict(new_current_states)
  
    X = []
    y = []

    # Now we need to enumerate our batches
    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

        # If not a terminal state, get new q from future states, otherwise set it to 0
        # almost like with Q Learning, but we use just part of equation here
        if not done:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q
        else:
            new_q = reward

        # Update Q value for given state
        current_qs = current_qs_list[index]
        current_qs[action] = new_q

        # And append to our training data
        X.append(current_state)
        y.append(current_qs)

    # Fit on all samples as one batch, log only on terminal state
    self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

    # Update target network counter every episode
    if terminal_state:
        self.target_update_counter += 1

    # If counter reaches set value, update target network with weights of main network
    if self.target_update_counter > UPDATE_TARGET_EVERY:
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
  
  # Queries main network for Q values given current observation space (environment state)
  def get_qs(self, state):
    return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]