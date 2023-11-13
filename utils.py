import cv2
import pickle
import random
import numpy as np
import copy
import networkx as nx
from torch import nn
import torch
from ale_py import ALEInterface, SDL_SUPPORT

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

# loads the possible state / dot coordinates from images (these were made by a seperate script which I've now replaced for a more trial-and-error approach)
def getCoordsInformation():
    global dot_coords
    global big_dot_coords
    global unobtained_dot_coords
    global unobtained_big_dot_coords
    global state_matrix
    global state_graph
    global current_level
    global tunnel_coords
    global prev_ghost_coords
    global prev_ghost_direction
    prev_ghost_coords = None
    prev_ghost_direction = [[0,0],[0,0],[0,0],[0,0]]
    
    if current_level < 2: # level 1
        tunnel_coords = [((12,50), (171,50)), ((12,98), (171,98))]
        possible_coords_img = 'data/possible_coords_level_1.png'
        dot_coords_img = 'data/dot_coords_level_1.png'
    else: # level 2
        tunnel_coords = [((2,62), (173,62)), ((2,158), (173,158))]
        possible_coords_img = 'data/possible_coords_level_2.png'
        dot_coords_img = 'data/dot_coords_level_2.png'
        
    state_matrix = np.ones((210,210))
    state_img = cv2.imread(possible_coords_img)
    for y in range(len(state_img)):
        for x in range(len(state_img[0])):
            if sum(state_img[y][x]) != 0: state_matrix[x][y] = 0.0
    state_graph = nx.grid_2d_graph(210, 210)
    walls = np.argwhere(state_matrix == 1)
    for pixel in walls: state_graph.remove_node(tuple(pixel))
    for (tunnel_x, tunnel_y), (other_tunnel_x, other_tunnel_y) in tunnel_coords:
        state_graph.add_edge((tunnel_x, tunnel_y), (other_tunnel_x, other_tunnel_y))
    dot_coords = []
    big_dot_coords = []
    dots_img = cv2.imread(dot_coords_img)
    for y in range(len(dots_img)):
        for x in range(len(dots_img[0])):
            if sum(dots_img[y][x]) > 0 and sum(dots_img[y][x]) == 255*3:
                dot_coords.append((x,y))
            elif sum(dots_img[y][x]) > 0:
                big_dot_coords.append((x,y))
    unobtained_dot_coords = copy.deepcopy(dot_coords)
    unobtained_big_dot_coords = copy.deepcopy(big_dot_coords)

## GLOBAL VARIABLES

# the current level the pacman is on
current_level = 1
last_num_dots_eaten = 0
getCoordsInformation()

### FOR THE ACTUAL AGENT

# returns the distance between two states using a weighted euclidean metric
def stateDistance(state1, state2):
    error = 0
    for i in range(state1):
        if state1[i] > 0 and state2[i] > 0:
            error_weight = 0.1
        else:
            error_weight = 2
        error += error_weight * (state1[i]-state2[i])**2
    return error

# given the location of the ghosts and the pacman, creates and returns a numpy array representing which tiles are "safe" for the pacman and which are not (safe meaning the pacman is closer to the square than the ghost is)
def buildSafeStateMatrix(pacman_x, pacman_y, djikstra_ghost_sources):
    getGraphInformation(pacman_x, pacman_y, djikstra_ghost_sources)
    global shortest_path_lengths_from_pacman
    global shortest_paths_from_pacman
    safe_state_matrix = copy.deepcopy(state_matrix) * 255
    ghost_dist_weight = 1.3 # make boundary between ghost and pacman at 60% instead of 50% of distance between them
    pacman_dist_weight = 1.7
    max_dist_for_ghost_avoidance = 200
    for x in range(len(state_matrix)):
        for y in range(len(state_matrix[0])):
            if state_matrix[x][y] == 1.0: continue
            # if closer to any ghost than it is to pacman set to 1 (obstacle)
            dist_to_pacman = shortest_path_lengths_from_pacman.get((x, y), float('inf'))
            min_dist_to_ghost = getShortestPathLengthToEnemy(x, y)
            if min_dist_to_ghost > max_dist_for_ghost_avoidance: min_dist_to_ghost = 9999 # if min_dist_to_ghost above a certain threshold than it is free space!!
            if min_dist_to_ghost*ghost_dist_weight <= dist_to_pacman*pacman_dist_weight: safe_state_matrix[x][y] = 255.0
    safe_state_matrix[pacman_x][pacman_y] = 255.0
    return np.uint8(safe_state_matrix)

def getGraphInformation(pacman_x, pacman_y, djikstra_ghost_sources):
    global shortest_path_lengths_from_pacman
    global shortest_paths_from_pacman
    global shortest_path_lengths_from_enemies
    global prev_ghost_coords
    global prev_ghost_direction
    global state_matrix
    
    # GET PACMAN DISTANCES
    shortest_path_lengths_from_pacman, shortest_paths_from_pacman = nx.single_source_dijkstra(state_graph, (pacman_x, pacman_y))
    
    # FOR ENEMIES, USE KNOWLEDGE THAT THEY CAN NEVER TURN AROUND 180 DEGREES
    if prev_ghost_coords is None:    
        prev_ghost_coords = djikstra_ghost_sources
    
    shortest_path_lengths_from_enemies = []    
    
    for i in range(4):
        # UPDATE CACHED GHOST DIRECTION
        dx = djikstra_ghost_sources[i][0] - prev_ghost_coords[i][0]
        if abs(dx) > 1: dx = int(dx / abs(dx))
        dy = djikstra_ghost_sources[i][1] - prev_ghost_coords[i][1]
        if abs(dy) > 1: dy = int(dy / abs(dy))

        if abs(dx) == 1 and abs(dy) == 1:
            # decide which was the previous position (no diagonal movements)
            if state_matrix[djikstra_ghost_sources[i][0]-dx][djikstra_ghost_sources[i][1]] == 0.0: 
                dy = 0
            else: 
                dx = 0
            prev_ghost_direction[i] = [dx,dy]
        elif not (dx == 0 and dy == 0):
            prev_ghost_direction[i] = [dx,dy]
        
        # UPDATE CLOSEST PATH LENGTHS FROM EACH ENEMY
        if prev_ghost_direction[i][0] != 0 or prev_ghost_direction[i][1] != 0:
            state_graph.remove_edge(djikstra_ghost_sources[i], (djikstra_ghost_sources[i][0]-prev_ghost_direction[i][0], djikstra_ghost_sources[i][1]-prev_ghost_direction[i][1]))
            shortest_path_lengths_from_enemies.append(nx.single_source_dijkstra_path_length(state_graph, djikstra_ghost_sources[i]))
            state_graph.add_edge(djikstra_ghost_sources[i], (djikstra_ghost_sources[i][0]-prev_ghost_direction[i][0], djikstra_ghost_sources[i][1]-prev_ghost_direction[i][1]))
        else:
            shortest_path_lengths_from_enemies.append(nx.single_source_dijkstra_path_length(state_graph, djikstra_ghost_sources[i]))
    print(prev_ghost_direction[0])
    prev_ghost_coords = djikstra_ghost_sources

def getShortestPathLengthToEnemy(x,y):
    global shortest_path_lengths_from_enemies
    shortest_path_length = float('inf')
    for path_lengths in shortest_path_lengths_from_enemies:
        shortest_path_length = min(path_lengths.get((x, y), float('inf')), shortest_path_length)
    return shortest_path_length

# constructs the state representation of the game using the ram, previous state, and previous action (previous state and previous action only used for momentum/to avoid oscillation of the pacman when it is in between two identically rated states)    
def buildStateFromRAM(ram, prev_state=None, prev_action=None):
    global current_level
    global unobtained_dot_coords
    global unobtained_big_dot_coords
    global last_num_dots_eaten
    global state_graph
    global shortest_path_lengths_from_pacman
    global shortest_paths_from_pacman
    
    # extract important info from ram
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
    
    # check if the level was passed using ram[119] (dots eaten count)
    if ram[119] == 0 and ram[119] != last_num_dots_eaten:
        current_level += 0.5
        getCoordsInformation()
    last_num_dots_eaten = ram[119]
    if current_level > 2.5:
        appendToFile("{},{}".format(player_x, player_y), "level_3_coords.csv")
    if current_level > 2.5: return [random.randint(0,100), random.randint(0,100), random.randint(0,100), random.randint(0,100)]
    
    # keep track of which dots the pacman has and hasn't eaten
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
    
    
    # hyperparameters: rewards associated with fruit, small dots, big dots, and momentum (prevents oscillation)
    fruit_val = 20
    dot_val = 2
    big_dot_val = 10
    momentum = 0.0
    # hyperparameters that define whether pacman is safe (far from ghosts) or in danger (close to ghosts)
    min_dist_for_rewards = 75
    min_dist_for_safety = 50
    
    ghost_coords = [(enemy_sue_x, enemy_sue_y), (enemy_inky_x, enemy_inky_y), (enemy_pinky_x, enemy_pinky_y), (enemy_blinky_x, enemy_blinky_y)]
    # initialize ghost coordinates for running of djikstra algorithm
    djikstra_ghost_sources = []
    for ghost_x, ghost_y in ghost_coords:
        if state_matrix[ghost_x][ghost_y] == 1.0:
            djikstra_ghost_sources.append((88,50))
            continue
        djikstra_ghost_sources.append((ghost_x,ghost_y))        
        
    # get matrix representing safe and unsafe grid coordinates
    safe_state_matrix = buildSafeStateMatrix(player_x, player_y, djikstra_ghost_sources)
    # debugging
    cv2.imshow('', safe_state_matrix.T)
    cv2.waitKey(1)

    # for each direction, calculate how much free space can be accessed in that direction
    # up
    if safe_state_matrix[player_x][player_y-1] == 255: available_space_up = 0
    else:
        up_safe_state_matrix = safe_state_matrix.copy()
        cv2.floodFill(up_safe_state_matrix,None,(player_y-1,player_x),64) # fill all connected "pixels" (grid coords) with value 64
        for (tunnel_x, tunnel_y), (other_tunnel_x, other_tunnel_y) in tunnel_coords: # for tunnels, continue filling on connected side of tunnel to account for ability to transfer between tunnel entries
                if up_safe_state_matrix[tunnel_x][tunnel_y] == 64:
                    cv2.floodFill(up_safe_state_matrix,None,(other_tunnel_y, other_tunnel_x),64)
                elif up_safe_state_matrix[other_tunnel_x][other_tunnel_y] == 64:
                    cv2.floodFill(up_safe_state_matrix,None,(tunnel_y, tunnel_x),64)
        # count number of pixels == 24 to count amount of free space in the up direction
        available_space_up = np.count_nonzero(up_safe_state_matrix == 64)
    # right
    if safe_state_matrix[player_x+1][player_y] == 255: available_space_right = 0
    else:
        right_safe_state_matrix = safe_state_matrix.copy()
        cv2.floodFill(right_safe_state_matrix,None,(player_y,player_x+1),64)
        for (tunnel_x, tunnel_y), (other_tunnel_x, other_tunnel_y) in tunnel_coords:
                if right_safe_state_matrix[tunnel_x][tunnel_y] == 64:
                    cv2.floodFill(right_safe_state_matrix,None,(other_tunnel_y, other_tunnel_x),64)
                elif right_safe_state_matrix[other_tunnel_x][other_tunnel_y] == 64:
                    cv2.floodFill(right_safe_state_matrix,None,(tunnel_y, tunnel_x),64)
        available_space_right = np.count_nonzero(right_safe_state_matrix == 64)
    # left
    if safe_state_matrix[player_x-1][player_y] == 255: available_space_left = 0
    else:
        left_safe_state_matrix = safe_state_matrix.copy()
        cv2.floodFill(left_safe_state_matrix,None,(player_y,player_x-1),64)
        for (tunnel_x, tunnel_y), (other_tunnel_x, other_tunnel_y) in tunnel_coords:
                if left_safe_state_matrix[tunnel_x][tunnel_y] == 64:
                    cv2.floodFill(left_safe_state_matrix,None,(other_tunnel_y, other_tunnel_x),64)
                elif left_safe_state_matrix[other_tunnel_x][other_tunnel_y] == 64:
                    cv2.floodFill(left_safe_state_matrix,None,(tunnel_y, tunnel_x),64)
        available_space_left = np.count_nonzero(left_safe_state_matrix == 64)
    # down
    if safe_state_matrix[player_x][player_y+1] == 255: available_space_down = 0
    else:
        down_safe_state_matrix = np.uint8(safe_state_matrix.copy())
        cv2.floodFill(down_safe_state_matrix,None,(player_y+1, player_x),64)
        for (tunnel_x, tunnel_y), (other_tunnel_x, other_tunnel_y) in tunnel_coords:
                if down_safe_state_matrix[tunnel_x][tunnel_y] == 64:
                    cv2.floodFill(down_safe_state_matrix,None,(other_tunnel_y, other_tunnel_x),64)
                elif down_safe_state_matrix[other_tunnel_x][other_tunnel_y] == 64:
                    cv2.floodFill(down_safe_state_matrix,None,(tunnel_y, tunnel_x),64)
        available_space_down = np.count_nonzero(down_safe_state_matrix == 64)
        
    
    
    min_dist_to_ghost = getShortestPathLengthToEnemy(player_x, player_y)
    safe = min_dist_to_ghost > min_dist_for_rewards
    danger = min_dist_to_ghost < min_dist_for_safety
    
    # check if state is ambiguous with softmax
    state = [available_space_up,
        available_space_right,
        available_space_left,
        available_space_down]
    softmaxed_rewards = nn.Softmax(dim=0)(torch.tensor(state, dtype=torch.float32))
    
    # if state is ambiguous and pacman is not in danger or pacman is safe, reset scores, so the decision will be made solely on rewards to be had    
    if safe or (not danger and max(softmaxed_rewards).item() < 0.65): # if pacman is "safe" and paths are not clearly better than each other, then make decision solely on points to be gained
        available_space_up = 0
        available_space_right = 0
        available_space_left = 0
        available_space_down = 0
    
    # calculate the shortest distance from every point in the map to pacman
    # use this to find the direction of the shortest path to the fruit and the distance from the fruit from the pacman
    direction_to_fruit = shortest_paths_from_pacman.get((fruit_x,fruit_y), [(-1,-1), (-1, -1)])[1]
    distance_to_fruit = shortest_path_lengths_from_pacman.get((fruit_x,fruit_y), np.inf)
    
    # only consider rewards if pacman is not in danger
    if not danger:
        if (player_x, player_y-1) == direction_to_fruit and up_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_up += fruit_val / distance_to_fruit
            
        if (player_x+1, player_y) == direction_to_fruit and right_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_right += fruit_val / distance_to_fruit
        
        if (player_x-1, player_y) == direction_to_fruit and left_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_left += fruit_val / distance_to_fruit
        
        if (player_x, player_y+1) == direction_to_fruit and down_safe_state_matrix[fruit_x][fruit_y] == 64: available_space_down += fruit_val / distance_to_fruit
        
    # account for small dot rewards
    for dot_x, dot_y in unobtained_dot_coords:
        
        # use this to find the djikstra shortest distance result from above to get the shortest path to the dot and the distance from the dot from the pacman
        direction_to_dot = shortest_paths_from_pacman.get((dot_x,dot_y), [(-1,-1), (-1, -1)])[1]
        distance_to_dot = shortest_path_lengths_from_pacman.get((dot_x,dot_y), np.inf)
        
        # only consider rewards if pacman is not in danger
        if not danger:
            if (player_x, player_y-1) == direction_to_dot and up_safe_state_matrix[dot_x][dot_y] == 64: available_space_up += dot_val / distance_to_dot
                
            if (player_x+1, player_y) == direction_to_dot and right_safe_state_matrix[dot_x][dot_y] == 64: available_space_right += dot_val / distance_to_dot
            
            if (player_x-1, player_y) == direction_to_dot and left_safe_state_matrix[dot_x][dot_y] == 64: available_space_left += dot_val / distance_to_dot
            
            if (player_x, player_y+1) == direction_to_dot and down_safe_state_matrix[dot_x][dot_y] == 64: available_space_down += dot_val / distance_to_dot
    
    # account for big dot rewards
    for dot_x, dot_y in unobtained_big_dot_coords:
        
        direction_to_dot = shortest_paths_from_pacman.get((dot_x,dot_y), [(-1,-1), (-1, -1)])[1]
        distance_to_dot = shortest_path_lengths_from_pacman.get((dot_x,dot_y), np.inf)
        
        # only consider rewards if pacman is not in danger
        if not danger:
            if (player_x, player_y-1) == direction_to_dot and up_safe_state_matrix[dot_x][dot_y] == 64: available_space_up += big_dot_val / distance_to_dot
                
            if (player_x+1, player_y) == direction_to_dot and right_safe_state_matrix[dot_x][dot_y] == 64: available_space_right += big_dot_val / distance_to_dot
            
            if (player_x-1, player_y) == direction_to_dot and left_safe_state_matrix[dot_x][dot_y] == 64: available_space_left += big_dot_val / distance_to_dot
            
            if (player_x, player_y+1) == direction_to_dot and down_safe_state_matrix[dot_x][dot_y] == 64: available_space_down += big_dot_val / distance_to_dot
    
    if prev_state is None: prev_state = [available_space_up, available_space_right, available_space_left, available_space_down]
    
    # apply momentum to state
    state = [available_space_up*(1-momentum) + momentum*prev_state[0],
            available_space_right*(1-momentum) + momentum*prev_state[1],
            available_space_left*(1-momentum) + momentum*prev_state[2],
            available_space_down*(1-momentum) + momentum*prev_state[3]]
    
    # use softmax to apply momentum to actions (i.e. if pacman was moving left and it is not sure where to go next, it should go left)
    softmaxed_rewards = nn.Softmax(dim=0)(torch.tensor(state, dtype=torch.float32))
    
    if max(softmaxed_rewards).item() < 0.65 and safe: # if the pacman is having trouble deciding a direction
        if prev_action == None:
            prev_action = torch.argmax(softmaxed_rewards).item()
        if softmaxed_rewards[prev_action-2] > 0.4:
            state[prev_action-2] *= 2
    return state

# creates and returns an instantiation of the ALE pacman environment with appropriate parameters
def makeEnvironment(render, seed=None):
    
    if seed is None: seed = random.randint(0, 9999)
    
    ale = ALEInterface()

    ale.setInt("random_seed", seed)

    if SDL_SUPPORT:
        ale.setBool("sound", True)
        ale.setBool("display_screen", render)

    ale.loadROM("data/MSPACMAN.BIN")
    ale.reset_game()
    
    return ale