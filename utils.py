from itertools import permutations
import cv2
import pickle
import gym
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

def shortest_angular_distance(angle1, angle2):
    # Calculate the angular difference
    angular_difference = angle2 - angle1

    # Ensure the result is within the range of -π to π
    while angular_difference > math.pi:
        angular_difference -= 2 * math.pi
    while angular_difference < -math.pi:
        angular_difference += 2 * math.pi

    return abs(angular_difference)

ghost_coordinates_combos = list(permutations([0,1,2,3]))
def stateDistance(state1, state2):
    pacman_dist_weight = 0.5
    ghost_dist_weight = 0.125/math.pi
    
    # euler distance between pacman coordinates
    pacman_dist = math.sqrt((state1[8] - state2[8])**2 + (state1[9] - state2[9])**2)
    # angle distance between ghosts
    
    # ghost_angle1 = shortest_angular_distance(math.atan2(state1[4], state1[0]),math.atan2(state2[4], state2[0]))
    # ghost_angle2 = shortest_angular_distance(math.atan2(state1[5], state1[1]),math.atan2(state2[5], state2[1]))
    # ghost_angle3 = shortest_angular_distance(math.atan2(state1[6], state1[2]),math.atan2(state2[6], state2[2]))
    # ghost_angle4 = shortest_angular_distance(math.atan2(state1[7], state1[3]),math.atan2(state2[7], state2[3]))
    # ghost_angle_dist = ghost_angle1+ghost_angle2+ghost_angle3+ghost_angle4
    
    ghost1_angle_1 = math.atan2(state1[4], state1[0])
    ghost2_angle_1 = math.atan2(state1[5], state1[1])
    ghost3_angle_1 = math.atan2(state1[6], state1[2])
    ghost4_angle_1 = math.atan2(state1[7], state1[3])
    
    ghost1_angle_2 = math.atan2(state2[4], state2[0])
    ghost2_angle_2 = math.atan2(state2[5], state2[1])
    ghost3_angle_2 = math.atan2(state2[6], state2[2])
    ghost4_angle_2 = math.atan2(state2[7], state2[3])
    
    ghost_angles_1 = [ghost1_angle_1,ghost2_angle_1,ghost3_angle_1,ghost4_angle_1]
    ghost_angle_dist = 99999999
    for ghost_coord_combo in ghost_coordinates_combos:
        ghost_angle_diff_1 = shortest_angular_distance(ghost1_angle_2, ghost_angles_1[ghost_coord_combo[0]])
        ghost_angle_diff_2 = shortest_angular_distance(ghost2_angle_2, ghost_angles_1[ghost_coord_combo[1]])
        ghost_angle_diff_3 = shortest_angular_distance(ghost3_angle_2, ghost_angles_1[ghost_coord_combo[2]])
        ghost_angle_diff_4 = shortest_angular_distance(ghost4_angle_2, ghost_angles_1[ghost_coord_combo[3]])

        ghost_angle_dist = min(ghost_angle_dist, ghost_angle_diff_1+ghost_angle_diff_2+ghost_angle_diff_3+ghost_angle_diff_4)

    return ghost_dist_weight*ghost_angle_dist + pacman_dist_weight*pacman_dist

    
def buildStateFromRAM(ram):
    ram = [int(r) for r in ram]
    enemy_sue_x = (ram[6] - 88) / 176
    enemy_inky_x = (ram[7] - 88) / 176
    enemy_pinky_x = (ram[8] - 88) / 176
    enemy_blinky_x = (ram[9] - 88) / 176
    enemy_sue_y = (74 - ram[12]) / 148
    enemy_inky_y = (74 - ram[13]) / 148
    enemy_pinky_y = (74 - ram[14]) / 148
    enemy_blinky_y = (74 - ram[15]) / 148
    player_x = (ram[10] - 88) / 176
    player_y = (74 - ram[16]) / 148
    
    # fruit_x = ram[11]
    # fruit_y = ram[17]
    
    return [enemy_sue_x-player_x,enemy_inky_x-player_x,enemy_pinky_x-player_x,enemy_blinky_x-player_x,enemy_sue_y-player_y,enemy_inky_y-player_y,enemy_pinky_y-player_y,enemy_blinky_y-player_y,player_x,player_y]
       
state_space = loadFromPickle("data/state_space.pkl")

def isAvailableAction(x,y,action):
    if action == 1: # up
        if (x,y+1) in state_space: return True
    elif action == 2: # right
        if (x+1,y) in state_space: return True
    elif action == 3: # left
        if (x-1,y) in state_space: return True
    else: # down
        if (x,y-1) in state_space: return True
    return False
        
def reward_fn(obs, next_obs, reward, action_taken):    
    obs = [int(o) for o in obs]
    next_obs = [int(o) for o in next_obs]
    
    past_num_lives = obs[123]
    next_num_lives = next_obs[123]
    
    if next_num_lives < past_num_lives: return -1
    elif reward >= 170: return -1
    else: reward = 0
    
    if not isAvailableAction(obs[10], obs[16], action_taken): return -0.05 #penalize moving towards walls
    
    if obs[10] == next_obs[10] and obs[16] == next_obs[16] and isAvailableAction(obs[10], obs[16], action_taken): return None # ignore times when pacman does not move even though it did a valid action
    
    ghost_1_dist = math.sqrt((obs[10] - obs[6])**2 + (obs[16] - obs[12])**2)
    ghost_2_dist = math.sqrt((obs[10] - obs[7])**2 + (obs[16] - obs[13])**2)
    ghost_3_dist = math.sqrt((obs[10] - obs[8])**2 + (obs[16] - obs[14])**2)
    ghost_4_dist = math.sqrt((obs[10] - obs[9])**2 + (obs[16] - obs[15])**2)
    old_min_ghost_dist = min(ghost_1_dist, ghost_2_dist, ghost_3_dist, ghost_4_dist)
    
    ghost_1_dist = math.sqrt((next_obs[10] - obs[6])**2 + (next_obs[16] - obs[12])**2)
    ghost_2_dist = math.sqrt((next_obs[10] - obs[7])**2 + (next_obs[16] - obs[13])**2)
    ghost_3_dist = math.sqrt((next_obs[10] - obs[8])**2 + (next_obs[16] - obs[14])**2)
    ghost_4_dist = math.sqrt((next_obs[10] - obs[9])**2 + (next_obs[16] - obs[15])**2)
    new_min_ghost_dist = min(ghost_1_dist, ghost_2_dist, ghost_3_dist, ghost_4_dist)
    
    if new_min_ghost_dist == 0: return 0
    ghost_dist_reward = 0.5 if (new_min_ghost_dist - old_min_ghost_dist) > 0 else -0.5

    # do not reward pacman for moving against the wall, even if the ghosts got further away
    # if moved_towards_wall: ghost_dist_reward = min(ghost_dist_reward, 0)
    
    reward += ghost_dist_reward
    
    return reward 

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



