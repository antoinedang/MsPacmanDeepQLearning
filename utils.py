import cv2
import pickle
import gym
import numpy as np
from astar.search import AStar

### FOR VISUALIZATION/DEBUGGING

def scaleImage(image, scale=4):
    return cv2.resize(image, (0,0), fx=scale, fy=scale) 
    
def saveImageToFile(image,filename="test_img.png"): cv2.imwrite(filename, image)

def saveToPickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
def loadFromPickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def appendToFile(text, filename):
    with open(filename, 'a') as f:
        f.write(text + '\n')
        
def loadBoolFromFile(filename):
    with open(filename, 'r+') as f:
        renderTrain, renderEval, renderRecording = f.readlines()[0].split(",")
        if renderEval == "True": renderEval = True
        else: renderEval = False
        if renderTrain == "True": renderTrain = True
        else: renderTrain = False
        if renderRecording == "True": renderRecording = True
        else: renderRecording = False
    return renderTrain, renderEval, renderRecording

### FOR THE ACTUAL AGENT

def stateDistance(state1, state2):
    error = 0
    for i in range(state1):
        error += (state1[i]-state2[i])**2
    return error

state_matrix = np.ones((210,210))
for x,y in loadFromPickle("data/state_space.pkl"):
    state_matrix[x][y] = 0.0
state_astar = AStar(state_matrix)

def findAStarDistanceInMap(x1,y1,x2,y2):
    if state_matrix[x1][y1] == 1.0: return 0
    if state_matrix[x2][y2] == 1.0: return 210
    return len(state_astar.search((x1,y1), (x2,y2)))
    
def buildStateFromRAM(ram):
    ram = [int(r) for r in ram]
    enemy_sue_x = ram[6]
    enemy_inky_x = ram[7]
    enemy_pinky_x = ram[8]
    enemy_blinky_x = ram[9]
    enemy_sue_y = ram[12]
    enemy_inky_y = ram[13]
    enemy_pinky_y = ram[14]
    enemy_blinky_y = ram[15]
    player_x = ram[10]
    player_y = ram[16]
    
    min_ghost_dist_down = 999999
    min_ghost_dist_right = 999999
    min_ghost_dist_left = 999999
    min_ghost_dist_up = 999999
    
    for enemy_x, enemy_y in [(enemy_sue_x, enemy_sue_y), (enemy_inky_x, enemy_inky_y), (enemy_pinky_x, enemy_pinky_y), (enemy_blinky_x, enemy_blinky_y)]:
        min_ghost_dist_down = min(min_ghost_dist_down, findAStarDistanceInMap(player_x, player_y+1, enemy_x, enemy_y))
        min_ghost_dist_right = min(min_ghost_dist_right, findAStarDistanceInMap(player_x+1, player_y, enemy_x, enemy_y))
        min_ghost_dist_left = min(min_ghost_dist_left, findAStarDistanceInMap(player_x-1, player_y, enemy_x, enemy_y))
        min_ghost_dist_up = min(min_ghost_dist_up, findAStarDistanceInMap(player_x, player_y-1, enemy_x, enemy_y))
    
    return [min_ghost_dist_down, min_ghost_dist_right, min_ghost_dist_left, min_ghost_dist_up]

def isAvailableAction(x,y,action):
    if action == 1: # up
        if state_matrix[x][y-1] == 0.0: return True
    elif action == 2: # right
        if state_matrix[x+1][y] == 0.0: return True
    elif action == 3: # left
        if state_matrix[x-1][y] == 0.0: return True
    else: # down
        if state_matrix[x][y+1] == 0.0: return True
    return False
        
def reward_fn(obs, next_obs, next_state, reward, action_taken):    
    prev_state = buildStateFromRAM(obs)
    
    past_num_lives = obs[123]
    next_num_lives = next_obs[123]
    
    if next_num_lives < past_num_lives: return -10
    elif reward >= 170: return -10
    else: reward = 0
    
    if obs[10] == next_obs[10] and obs[16] == next_obs[16] and isAvailableAction(obs[10], obs[16], action_taken): return None # ignore times when pacman does not move even though it did a valid action
    elif obs[10] == next_obs[10] and obs[16] == next_obs[16] and not isAvailableAction(obs[10], obs[16], action_taken): return -0.5 # punish running into walls
    
    min_prev_state = 99999
    
    for i in range(len(prev_state)):
        if prev_state[i] == 0: continue
        min_prev_state = min(min_prev_state, prev_state[i])
    
    dist_change = next_state[prev_state.index(min_prev_state)] - min_prev_state
    
    return dist_change

def makeEnvironment():
    return gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')

# EASY
# 20. graph or table showing agent performance, either as a function of game score or game time (before death of game agent) as a function of RL time (e.g., number of games played)
# 30. as above, including analysis of effects of game events on agent behaviour and strategies (e.g., "enemy" avoidance) AND explanation of results over trials with multiple seeds to demonstrate generalization of learning
# -> get recording of 3 games after every epoch for analysis

# EASY IF WE CACHE SEEN vs. UNSEEN STATES
# 10. results provided for at least 2 different exploration functions (i.e., weighting or N[s,a] in optimistic prior calculation) and meaningful discussion regarding consequences to behaviour of game agent
# -> random actions
# -> new (unseen) state/actions are weighted highly

# 10 provides expression for optimistic prior for (state, action) pairs with clear explanation of how agent chose action at each step and convincing rationale for the approach taken
# -> for unseen states and which are not close to seen states, value is very high (value inversely proportional to distance to state)

# 10. results provided for at least 3 different generalization approaches (i.e., choice of components of the distance metric) and meaningful discussion regarding consequences to behaviour of game agent
# -> 3 different choices of distance metrics (stupid, smart with some components, smart with different components)
# -> train and analyze each

# 10. provides an expression for distance metric between two states and describes state representation used by the RL agent, also includes rationale for choice of components of the distance metric (Mahadevan)
# -> easy

# State Representation: position of each enemy, position of pacman
    
# State Distance: (and components)

# Action Representation: continue direction, turn left, turn right, turn around

# Exploration Functions: random actions, high value for unexplored state/action pairs



