from itertools import permutations
import numpy as np
import cv2
import pickle
import copy
import math

### FOR VISUALIZATION/DEBUGGING

def scaleImage(image, scale=2):
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

### FOR THE ACTUAL AGENT
        
ghost_coordinates_combos = list(permutations([0,1,2,3]))
def stateDistance(state1, state2):
    # euler distance between pacman coordinates
    ghost_coordinates_x = [state1[0], state1[1], state1[2], state1[3]]
    ghost_coordinates_y = [state1[4], state1[5], state1[6], state1[7]]
    
    pacman_dist_weight = 1
    ghost_dist_weight = 0.25
    
    pacman_dist = (state1[8] - state2[8])**2 + (state1[9] - state2[9])**2
    return pacman_dist
    min_ghost_dist = 100000000
    
    for ghost_coordinates_combo in ghost_coordinates_combos:
        for i in ghost_coordinates_combo:
            state1[i] = ghost_coordinates_x[i]
            state1[i+4] = ghost_coordinates_y[i]
        ghost_1_dist = (state1[0] - state2[0])**2 + (state1[4] - state2[4])**2
        ghost_2_dist = (state1[1] - state2[1])**2 + (state1[5] - state2[5])**2
        ghost_3_dist = (state1[2] - state2[2])**2 + (state1[6] - state2[6])**2
        ghost_4_dist = (state1[3] - state2[3])**2 + (state1[7] - state2[7])**2
        ghost_dist = ghost_1_dist + ghost_2_dist + ghost_3_dist + ghost_4_dist
        min_ghost_dist = min(ghost_dist, min_ghost_dist)
    # smallest angle difference between pairs enemy coordinates
    # smallest euler distance between pairs of enemy coordinates
    return ghost_dist_weight*min_ghost_dist + pacman_dist_weight*pacman_dist
    
dot_coordinates = loadFromPickle("data/dot_coordinates.pkl")
    
def buildStateFromRAM(ram):
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
    # fruit_x = ram[11]
    # fruit_y = ram[17]
    
    return [enemy_sue_x,enemy_inky_x,enemy_pinky_x,enemy_blinky_x,enemy_sue_y,enemy_inky_y,enemy_pinky_y,enemy_blinky_y,player_x,player_y]
        
        
def reward_fn(obs, next_obs, reward):    
    past_num_lives = obs[123]
    next_num_lives = next_obs[123]
    
    past_pacman_x = obs[10]
    next_pacman_x = next_obs[10]
    past_pacman_y = obs[16]
    next_pacman_y = next_obs[16]
    
    if reward >= 200: reward = -100
    
    if past_pacman_x == next_pacman_x and past_pacman_y == next_pacman_y:
        reward += -1
    
    if next_num_lives < past_num_lives:
        reward -= 100
        
    return reward    


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



