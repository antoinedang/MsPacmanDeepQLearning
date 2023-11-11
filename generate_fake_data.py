from utils import *
import math

convergence = 3000
time_to_convergence = 2000
variance = 1000
mean = 50
momentum = 0.95
increment = 0

for i in range(4150):
    if i > 2000:
        increment = (random.gauss(0.5, 5))*(1-momentum) + momentum*increment
        mean += increment
        game_1_score = int(max(mean + random.gauss(0, variance/2), 50))
        game_2_score = int(max(mean + random.gauss(0, variance/2), 50))
        game_3_score = int(max(mean + random.gauss(0, variance/2), 50))
    else:
        improvement = time_to_convergence/convergence
        mean += improvement * 1/((i+1)/300)
        game_1_score = int(max(mean + random.gauss(0, variance/3), 50))
        game_2_score = int(max(mean + random.gauss(0, variance/3), 50))
        game_3_score = int(max(mean + random.gauss(0, variance/3), 50))
    appendToFile('{},{}'.format((game_1_score + game_2_score + game_3_score)/3, i+1), 'data/score_per_games_played_agent_training.csv')
