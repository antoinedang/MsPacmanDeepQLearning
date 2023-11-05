import numpy as np
import cv2
import pickle

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


def stateDistance(state1, state2):
    pass

ram_mappings = dict(enemy_sue_x=6,
                     enemy_inky_x=7,
                     enemy_pinky_x=8,
                     enemy_blinky_x=9,
                     enemy_sue_y=12,
                     enemy_inky_y=13,
                     enemy_pinky_y=14,
                     enemy_blinky_y=15,
                     player_x=10,
                     player_y=16,
                     fruit_x=11,
                     fruit_y=17,
                     ghosts_count=19,
                     player_direction=56,
                     dots_eaten_count=119,
                     player_score=120,
                     num_lives=123)
    
possible_coordinates = loadFromPickle("data/possible_coordinates.pkl")
    
def buildStateFromRAM(ram, prev_state):
    enemy_sue_x = 0
    enemy_sue_y =0
    enemy_pinky_x = 0
    enemy_blinky_x = 0
    enemy_sue_y =0 
    enemy_inky_y =0 
    enemy_pinky_y = 0 
    enemy_blinky_y = 0
    player_x = 0
    player_y = 0
    current_coords = (player_x, player_y)
    current_coords_idx = possible_coordinates.index(current_coords)
    prev_state[0][current_coords_idx] = 1
    
    new_state = np.array()
        


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

# State Representation: position of each enemy, position of pacman, position of each unobtained dot
    
# State Distance: (and components)

# Action Representation: continue direction, turn left, turn right, turn around

# Exploration Functions: random actions, high value for unexplored state/action pairs



