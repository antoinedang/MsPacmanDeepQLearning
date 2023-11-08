import numpy as np
import cv2
import keyboard
from utils import *

def analyzeStateSpace(filename):
    possible_coordinates = []
    small_dots = []
    big_dots = []
    with open(filename, "r") as f:
        for line in f.readlines():
            pacman_x, pacman_y, reward = line[:-1].split(",")
            possible_coordinates.append((int(pacman_x), int(pacman_y)))
            if reward not in ["0.0", "10.0", "50.0"]: print("Not sure about {} {} : worth {} points".format(pacman_x, pacman_y, reward))
            else:
                reward = float(reward)
                while reward >= 100: reward -= 100
                if reward == 60:
                    small_dots.append((int(pacman_x), int(pacman_y)))
                    big_dots.append((int(pacman_x), int(pacman_y)))
                elif reward == 10.0: small_dots.append((int(pacman_x), int(pacman_y)))
                elif reward == 50.0: big_dots.append((int(pacman_x), int(pacman_y)))
    possible_coordinates = set(possible_coordinates)
    small_dots = set(small_dots)
    big_dots = set(big_dots)
    
    test_img = np.zeros((210, 210, 3))
    
    for coordinate in possible_coordinates:
        test_img[coordinate[1]][coordinate[0]] = np.array([255,255,255])
        
    # for coordinate in small_dots:
    #     test_img[coordinate[1]][coordinate[0]] = np.array([255,0,0])
        
    # for coordinate in big_dots:
    #     test_img[coordinate[1]][coordinate[0]] = np.array([0,255,0])
    
    saveToPickle("data/state_space.pkl", list(possible_coordinates))
    
    cv2.imshow('',test_img)
    saveImageToFile(test_img,filename="data/possible_coords_img.png")
    cv2.waitKey(0)

def generateDotCoordinates(img_filename):
    dot_coords_img = cv2.imread(img_filename)
    
    dot_coords = []
    
    for y in range(len(dot_coords_img)):
        for x in range(len(dot_coords_img[0])):
            if sum(dot_coords_img[y][x]) != 0:
                dot_coords.append((x,y))
    
    saveToPickle("data/dot_coordinates.pkl", list(set(dot_coords)))
    

if __name__ == '__main__':
    
    ## UNCOMMENT THIS TO CREATE A PICKLE OF ALL DOT COORDINATES FROM AN IMAGE
    
    # generateDotCoordinates("data/dot_coords.png")    
    
    ## UNCOMMENT BELOW TO GENERATE ALL POSSIBLE COORDINATES
    
    # exit()
    
    while True:
    
        analyzeStateSpace("data/possible_coordinates.csv")
        
        env = makeEnvironment()

        obs = env.reset()
        total_reward = 0
        # saveImageToFile(env.render())
        
        while True:
            game_img = env.render()
            saveImageToFile(game_img,filename="data/game_img.png")
            cv2.imshow('',scaleImage(game_img))
            cv2.waitKey(1)
            if keyboard.is_pressed('left'):
                action = 3
            elif keyboard.is_pressed('right'):
                action = 2
            elif keyboard.is_pressed('down'):
                action = 4
            elif keyboard.is_pressed('up'):
                action = 1
            else:
                action = 0
            
            obs, reward, done, _, info = env.step(action)
            pacman_x, pacman_y = env.unwrapped.ale.getRAM()[10], env.unwrapped.ale.getRAM()[16]
            appendToFile("{},{},{}".format(pacman_x, pacman_y, reward), "data/possible_coordinates.csv")
            total_reward += reward
            if done:
                break
        env.close()
        