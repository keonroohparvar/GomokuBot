"""
This script contains helper functions that help our model.

Author: Keon Roohparvar
Date: April 6, 2022

"""

import numpy as np

# Helper function for Heuristic
def find_length_in_row(row, player):
  counts = [0, 0, 0, 0, 0]
  i = 0
  while i < len(row):
    curr_count = 0
    if row[i] == player:
      j = i + 1
      while j < len(row) and row[j] == player:
        curr_count += 1
        j += 1
      
      counts[curr_count] += counts[curr_count] + 1
      i = j - 1
    
    i += 1
  return counts[1:]

def find_score(board, player):
  # This is [num_2_components, num_3_components, ...]
  # Note that we are not counting number of 1 components as that is not helpful

  # A LOT OF BOOKKEEPING
  row_board = np.array(board)
  col_board = row_board.T 

  diags = [row_board[::-1,:].diagonal(i) for i in range(-row_board.shape[0]+1,row_board.shape[1])]
  diags.extend(row_board.diagonal(i) for i in range(row_board.shape[1]-1,-row_board.shape[0],-1))
  r_diag_board = [n.tolist() for n in diags][1:-1]

  diags = [col_board[::-1,:].diagonal(i) for i in range(-col_board.shape[0]+1,col_board.shape[1])]
  diags.extend(col_board.diagonal(i) for i in range(col_board.shape[1]-1,-col_board.shape[0],-1))
  l_diag_board = [n.tolist() for n in diags][1:-1]

  rows = [find_length_in_row(n, player) for n in row_board]
  cols = [find_length_in_row(n, player) for n in col_board]
  ldiag = [find_length_in_row(n, player) for n in l_diag_board]
  rdiag = [find_length_in_row(n, player) for n in r_diag_board]

  # Sum all scores in all counts
  counts = [0, 0, 0, 0]

  for a, b, c, d in rows:
    counts[0] += a
    counts[1] += b
    counts[2] += c
    counts[3] += d
  
  for a, b, c, d in cols:
    counts[0] += a
    counts[1] += b
    counts[2] += c
    counts[3] += d

  for a, b, c, d in ldiag:
    counts[0] += a
    counts[1] += b
    counts[2] += c
    counts[3] += d
  
  for a, b, c, d in rdiag:
    counts[0] += a
    counts[1] += b
    counts[2] += c
    counts[3] += d
  
  return counts

def find_reward(env, color, mults, opponent_mult):
  player_color = env.player_color
  player = 1 if player_color is 'black' else 2
  state = env.state.board.board_state

  player_reward = find_score(state, player)
  computer_reward = find_score(state, 2 if player_color is 'black' else 1)

  # If player has won, just return 100,000
  if player_reward[3] > 0:
    return 100000
  
  elif computer_reward[3] > 0:
    return -100000

  player_sum = sum([(player_reward[i] * mults[i]) for i in range(len(player_reward))])
  computer_sum = sum([(computer_reward[i] * mults[i] * opponent_mult) for i in range(len(computer_reward))])


  return player_sum - computer_sum

# Finds maximum Q Value predicted 
def find_max_Q(qs, valid_spaces):
  max_val = qs[valid_spaces[0]]
  max_idx = valid_spaces[0]
  for i in valid_spaces[1:]:
    if qs[i] > max_val:
      max_val = qs[i]
      max_idx = i
  
  return max_idx