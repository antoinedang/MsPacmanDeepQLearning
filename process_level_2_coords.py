from utils import *

possible_coords_img = 'data/possible_coords_level_2.png'
dot_coords_img = 'data/dot_coords_level_2.png'

coords = []

with open("level_2_coords.csv", 'r') as f:
    for line in f.readlines():
        x, y = line.split(",")
        coords.append((int(x), int(y)))


state_img = cv2.imread(possible_coords_img)
dots_img = cv2.imread(possible_coords_img)
for y in range(len(state_img)):
    for x in range(len(state_img[0])):
        if (x,y) in coords and sum(state_img[y][x]) == 0:
            print("Missing coordinate! {},{}".format(x,y))
        if sum(state_img[y][x]) == 0 and sum(dots_img[y][x]) != 0:
            print("Invalid coordinate in dots image! {},{}".format(x,y))
        # if (x,y) not in coords and sum(state_img[y][x]) != 0:
        #     print("Invalid coordinate in image! {},{}".format(x,y))
        # if (x,y) not in coords and sum(dots_img[y][x]) != 0:
        #     print("Invalid dot coordinate in image! {},{}".format(x,y))

print("Done.")