import cv2
import pickle
import gym
import numpy as np
from astar.search import AStar
import copy
import math

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
for x,y in [(11,98), (10,98), (9,98), (172,98), (173,98), (174,98), (11,50), (10,50), (9,50), (172,50), (173,50), (174,50)]: # add tunnel entry points to coordinate space
    state_matrix[x][y] = 0.0
state_astar = AStar(state_matrix)

def buildSafeStateMatrix(pacman_x, pacman_y, ghost_coords):
    safe_state_matrix = copy.deepcopy(state_matrix) * 255
    ghost_dist_weight = 1.3 # make boundary between ghost and pacman at 60% instead of 50% of distance between them
    pacman_dist_weight = 1.7
    for x in range(len(state_matrix)):
        for y in range(len(state_matrix[0])):
            if state_matrix[x][y] == 1.0: continue
            # if closer to any ghost than it is to pacman set to 1 (obstacle)
            # dist_to_pacman = findAStarDistanceInMap(pacman_x, pacman_y, x, y)
            dist_to_pacman = math.sqrt((pacman_x-x)**2 + (pacman_y-y)**2)
            min_dist_to_ghost = 9999
            for ghost_x, ghost_y in ghost_coords:
                min_dist_to_ghost = min(min_dist_to_ghost, math.sqrt((ghost_x-x)**2 + (ghost_y-y)**2))
                # min_dist_to_ghost = min(min_dist_to_ghost, findAStarDistanceInMap(ghost_x, ghost_y, x, y))
            if min_dist_to_ghost*ghost_dist_weight <= dist_to_pacman*pacman_dist_weight: safe_state_matrix[x][y] = 255.0
    safe_state_matrix[pacman_x][pacman_y] = 255.0
    return np.uint8(safe_state_matrix)

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
    
    safe_state_matrix = buildSafeStateMatrix(player_x, player_y, [(enemy_sue_x, enemy_sue_y), (enemy_inky_x, enemy_inky_y), (enemy_pinky_x, enemy_pinky_y), (enemy_blinky_x, enemy_blinky_y)])
    
    if safe_state_matrix[player_x][player_y-1] == 255: available_space_up = 0
    else:
        up_safe_state_matrix = safe_state_matrix.copy()
        cv2.floodFill(up_safe_state_matrix,None,(player_y-1,player_x),64)
        available_space_up = np.count_nonzero(up_safe_state_matrix == 64)
        
    if safe_state_matrix[player_x+1][player_y] == 255: available_space_right = 0
    else:
        right_safe_state_matrix = safe_state_matrix.copy()
        cv2.floodFill(right_safe_state_matrix,None,(player_y,player_x+1),64)
        available_space_right = np.count_nonzero(right_safe_state_matrix == 64)
        
    if safe_state_matrix[player_x-1][player_y] == 255: available_space_left = 0
    else:
        left_safe_state_matrix = safe_state_matrix.copy()
        cv2.floodFill(left_safe_state_matrix,None,(player_y,player_x-1),64)
        available_space_left = np.count_nonzero(left_safe_state_matrix == 64)
        
    if safe_state_matrix[player_x][player_y+1] == 255: available_space_down = 0
    else:
        down_safe_state_matrix = np.uint8(safe_state_matrix.copy())
        cv2.floodFill(down_safe_state_matrix,None,(player_y+1, player_x),64)
        available_space_down = np.count_nonzero(down_safe_state_matrix == 64)
    
    return [available_space_up, available_space_right, available_space_left, available_space_down]

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



