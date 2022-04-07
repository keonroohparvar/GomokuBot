# GomokuBot
This software is an implementation of a reinforcement learning model that learn how to play Gomoku (five-in-a-row). It started as a group project for my _CSC 480 - Artificial Intelligence_ class, and it took a bit of inspiration from the [tutorial](https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/) of a reinforcement learning approach to beating the _Blob_ game. The course ended in March 2022, and all work done after that is my own development - I am continuing because I felt that this project initially had a lot of shortcominings, and I am excited to make it more rigorous. 

## Repository File Structure

All of the developed code for this project is located in the _src/_ folder. In there, you can find the following files:

### environment.py

This file serves to modularize all of the code that involves the Python [Gym](https://gym.openai.com) code; it is the framework that keeps track of our board, and allows us to make moves. It is heavily reliant on the Python [gym-gomoku API](https://github.com/rockingdingo/gym-gomoku). 

### legacy_file.py
This is the original Google Colab file that we were initially running all of our development off of. It got refactored into all of the files found in _src/_, and its current purpose is just as a failsafe in case of any fatal errors encountered that I cannot figure out due to the connections between the newly modularized files (which hopefully never happens, but who knows?!?!?!). 

### model_architecture.py
This file contains our Deep Q-Learning agent implementation. It was originally a reimplementation of the tutorial mentioned above on the _Blob_ game, but it was heavily adapted to account for the complexity of Gomoku.

### model_helper_functions.py
This script contains helper logic that mainly helps with the interface between the Deep Q-Learning agent and the Gomoku environment.

### modified_tensorboard.py
This tensorboard allows us to concatenate multiple training sessions into one training log, which is essential for viewing the logs that our numerous training attempts will make.

### train.py
This file contains all the logic of training our Deep Q-Learning agent and incrementally saving its model weights out to a save file. This file also contains the hyperparameters related to training, including the number of training episodes, the weights for our heuristic, our epsilon value, and our epsilon decay value.
