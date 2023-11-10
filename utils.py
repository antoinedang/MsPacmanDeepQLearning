import cv2
import pickle
import gym
import numpy as np
import copy
import math
import networkx as nx

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

## GLOBAL VARIABLES

state_matrix = np.ones((210,210))
for x,y in loadFromPickle("data/state_space.pkl"):
    state_matrix[x][y] = 0.0
for x,y in [(11,98), (10,98), (9,98), (8,98), (7,98), (6,98), (5,98), (4,98), (3,98), (2,98), (1,98), (0,98),
            (172,98), (173,98), (174,98), (175,98), (176,98), (177,98), (178,98), (179,98), (180,98), (181,98),
            (11,50), (10,50), (9,50), (8,50), (7,50), (6,50), (5,50), (4,50), (3,50), (2,50), (1,50), (0,50),
            (172,50), (173,50), (174,50), (175,50), (176,50), (177,50), (178,50), (179,50), (180,50), (181,50)]: # add tunnel entry points to coordinate space
    state_matrix[x][y] = 0.0
rows, cols = state_matrix.shape
state_graph = nx.grid_2d_graph(rows, cols)
# Remove wall pixels from the graph
walls = np.argwhere(state_matrix == 1)
for pixel in walls: state_graph.remove_node(tuple(pixel))

dot_coords = loadFromPickle("data/dot_coordinates.pkl")
big_dot_coords = [(18,12), (158,12), (18,142), (158,142)]
unobtained_dot_coords = copy.deepcopy(dot_coords)
unobtained_big_dot_coords = copy.deepcopy(big_dot_coords)

### FOR THE ACTUAL AGENT

def stateDistance(state1, state2):
    error = 0
    for i in range(state1):
        error += (state1[i]-state2[i])**2
    return error

def buildSafeStateMatrix(pacman_x, pacman_y, ghost_coords):
    shortest_path_lengths_from_pacman = nx.single_source_dijkstra_path_length(state_graph, (pacman_x, pacman_y))
    djikstra_ghost_sources = set()
    for ghost_x, ghost_y in ghost_coords:
        if state_matrix[ghost_x][ghost_y] == 1.0:
            djikstra_ghost_sources.add((88,50))
            continue
        djikstra_ghost_sources.add((ghost_x,ghost_y))
    shortest_path_lengths_from_enemies = nx.multi_source_dijkstra_path_length(state_graph, djikstra_ghost_sources)
    safe_state_matrix = copy.deepcopy(state_matrix) * 255
    ghost_dist_weight = 1.3 # make boundary between ghost and pacman at 60% instead of 50% of distance between them
    pacman_dist_weight = 1.7
    max_dist_for_ghost_avoidance = 150
    for x in range(len(state_matrix)):
        for y in range(len(state_matrix[0])):
            if state_matrix[x][y] == 1.0: continue
            # if closer to any ghost than it is to pacman set to 1 (obstacle)
            dist_to_pacman = shortest_path_lengths_from_pacman.get((x, y), float('inf'))
            min_dist_to_ghost = shortest_path_lengths_from_enemies.get((x, y), float('inf'))
            # # dist_to_pacman = math.sqrt((pacman_x-x)**2 + (pacman_y-y)**2)
            # min_dist_to_ghost = 9999
            # for ghost_x, ghost_y in ghost_coords:
                # min_dist_to_ghost = min(min_dist_to_ghost, math.sqrt((ghost_x-x)**2 + (ghost_y-y)**2))
            if min_dist_to_ghost > max_dist_for_ghost_avoidance: min_dist_to_ghost = 9999 # if min_dist_to_ghost above a certain threshold than it is free space!!
            if min_dist_to_ghost*ghost_dist_weight <= dist_to_pacman*pacman_dist_weight: safe_state_matrix[x][y] = 255.0
    safe_state_matrix[pacman_x][pacman_y] = 255.0
    return np.uint8(safe_state_matrix)
    
def buildStateFromRAM(ram, prev_state=None):
    global unobtained_dot_coords
    global unobtained_big_dot_coords
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
    fruit_x = ram[11]
    fruit_y = ram[17]
    
    if ram[119] == 0:
        unobtained_dot_coords = copy.deepcopy(dot_coords)
        unobtained_big_dot_coords = copy.deepcopy(big_dot_coords)
    
    dots_eaten = []
    for dot_x, dot_y in unobtained_dot_coords:
        if abs(player_x - dot_x) < 2 and abs(player_y - dot_y) < 2:
            dots_eaten.append((dot_x,dot_y))
    big_dots_eaten = []
    for dot_x, dot_y in unobtained_big_dot_coords:
        if abs(player_x - dot_x) < 2 and abs(player_y - dot_y) < 2:
            big_dots_eaten.append((dot_x,dot_y))
    
    for dot in dots_eaten: unobtained_dot_coords.remove(dot)
    for dot in big_dots_eaten: unobtained_big_dot_coords.remove(dot)
    
    fruit_val = 10
    dot_val = 2
    big_dot_val = 2 #added to dot_val
    
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
    
    _, shortest_paths_from_pacman = nx.single_source_dijkstra(state_graph, (player_x, player_y))
    
    direction_to_fruit = shortest_paths_from_pacman.get((fruit_x,fruit_y), [(-1,-1), (-1, -1)])[1]
    
    if available_space_up != 0:
        if (player_x, player_y-1) == direction_to_fruit and up_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_up += fruit_val
            
    if available_space_right != 0:
        if (player_x+1, player_y) == direction_to_fruit and right_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_right += fruit_val
        
    if available_space_left != 0:
        if (player_x-1, player_y) == direction_to_fruit and left_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_left += fruit_val
        
    if available_space_down != 0:
        if (player_x, player_y+1) == direction_to_fruit and down_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_down += fruit_val   
        
    for dot_x, dot_y in unobtained_dot_coords:
        
        direction_to_dot = shortest_paths_from_pacman.get((dot_x,dot_y), [(-1,-1), (-1, -1)])[1]
        
        if available_space_up != 0:
            if (player_x, player_y-1) == direction_to_dot and up_safe_state_matrix[dot_x][dot_y] == 64: available_space_up += fruit_val
                
        if available_space_right != 0:
            if (player_x+1, player_y) == direction_to_dot and right_safe_state_matrix[dot_x][dot_y] == 64: available_space_right += fruit_val
            
        if available_space_left != 0:
            if (player_x-1, player_y) == direction_to_dot and left_safe_state_matrix[dot_x][dot_y] == 64: available_space_left += fruit_val
            
        if available_space_down != 0:
            if (player_x, player_y+1) == direction_to_dot and down_safe_state_matrix[dot_x][dot_y] == 64: available_space_down += fruit_val   
    
    for dot_x, dot_y in unobtained_big_dot_coords:
        
        direction_to_dot = shortest_paths_from_pacman.get((dot_x,dot_y), [(-1,-1), (-1, -1)])[1]
        
        if available_space_up != 0:
            if (player_x, player_y-1) == direction_to_dot and up_safe_state_matrix[dot_x][dot_y] == 64: available_space_up += fruit_val
                
        if available_space_right != 0:
            if (player_x+1, player_y) == direction_to_dot and right_safe_state_matrix[dot_x][dot_y] == 64: available_space_right += fruit_val
            
        if available_space_left != 0:
            if (player_x-1, player_y) == direction_to_dot and left_safe_state_matrix[dot_x][dot_y] == 64: available_space_left += fruit_val
            
        if available_space_down != 0:
            if (player_x, player_y+1) == direction_to_dot and down_safe_state_matrix[dot_x][dot_y] == 64: available_space_down += fruit_val   
    
    
    momentum = 0.0
    if prev_state is None: prev_state = [available_space_up, available_space_right, available_space_left, available_space_down]
    
    return [available_space_up*(1-momentum) + momentum*prev_state[0],
            available_space_right*(1-momentum) + momentum*prev_state[1],
            available_space_left*(1-momentum) + momentum*prev_state[2],
            available_space_down*(1-momentum) + momentum*prev_state[3]]

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



