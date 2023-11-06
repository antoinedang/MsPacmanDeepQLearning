import matplotlib.pyplot as plt
from scipy.stats import linregress
import time

while True:
    scores = []
    n_games_played = []

    with open("data/score_per_games_played.csv", "r+") as f:
        for line in f.readlines():
            if "," in line:
                score, games_played = line.split(",")
                games_played = games_played[:-1]
            scores.append(float(score))
            n_games_played.append(int(games_played))
        
    plt.plot(n_games_played, scores, marker='', color='b', linestyle='-', label='Scores vs. Games Played')

    slope, intercept, r_value, p_value, std_err = linregress(n_games_played, scores)
    trendline = [slope * n + intercept for n in n_games_played]
    plt.plot(n_games_played, trendline, color='r', label=f'Trendline (R-squared={r_value**2:.4f})')

    plt.xlabel('Games Played')
    plt.ylabel('Score')
    plt.title('Score vs. Games Played')
    plt.grid(True)
    plt.legend()
    plt.show()

    # plt.close()