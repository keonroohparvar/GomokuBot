"""
This file contains all of the logic regarding the Gomoku Environment.

Author: Keon Roohparvar
Date: April 6, 2022

"""

import gym
import gym_gomoku
from gym.envs.registration import register

def get_env(size, random=False):
    # Register new 15x15 environment
    register(
        id=f'Gomoku{size}x{size}-v1',
        entry_point='gym_gomoku.envs:GomokuEnv',
        kwargs={
            'player_color': 'black',
            'opponent': 'beginner', # beginner opponent policy has defend and strike rules
            'board_size': 15,
        },
        nondeterministic=True,
    )

    # Register Random Gomoku Env
    register(
        id='Gomoku{size}x{size}-v0',
        entry_point='gym_gomoku.envs:GomokuEnv',
        kwargs={
            'player_color': 'black',
            'opponent': 'random',
            'board_size': 15,
        },
        nondeterministic=True,
    )

    ENV_RANDOM = False

    if ENV_RANDOM:
        return gym.make('Gomoku15x15-v0')
    else:
        return gym.make('Gomoku15x15-v1')
