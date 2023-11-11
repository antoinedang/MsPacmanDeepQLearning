from agent import *
import warnings
from utils import *
import random

# Suppress all warnings
warnings.filterwarnings("ignore")

def log(score, games_played):
    # log to CSV file
    appendToFile("{},{}".format(score, games_played), "data/score_per_games_played.csv")
    pass

def train(episodes):
    ale = makeEnvironment(render=render_train)
    print("Epoch progress: 0%                         ", end='\r')
    for ep in range(episodes):
        ale.reset_game()
        random.seed(random.randint(0, 10000))
        obs = ale.getRAM()
        state = buildStateFromRAM(obs)
        action = None
        next_is_random = 0
        lives_left = 3
        while not ale.game_over():
            if next_is_random > 0:
                next_is_random -= 1
            else:
                action, next_is_random = agent.getAction(state)
            _ = ale.act(action)
            next_obs = ale.getRAM()
            state = buildStateFromRAM(next_obs, state, action)
            agent.update(state, action, obs)
            if next_obs[123] < obs[123]:
                lives_left -= 1
                print("Epoch progress: {}%...                       ".format(100*(3*ep+(3-lives_left))/(3*episodes)), end='\r')
            obs = next_obs
        agent.games_played += 1

def evaluate(games):
    p_random_action = agent.p_random_action
    explore_unseen_states = agent.explore_unseen_states
    agent.p_random_action = 0
    agent.explore_unseen_states = False
    ale = makeEnvironment(render=render_eval)
    print("Evaluation progress: 0%                  ", end='\r')
    total_reward = 0
    for ep in range(games):
        ale.reset_game()
        random.seed(random.randint(0, 10000))
        lives_left = 3
        state = None
        action = None
        while not ale.game_over():
            obs = ale.getRAM()
            state = buildStateFromRAM(obs, state, action)
            action, _ = agent.getAction(state)
            total_reward += ale.act(action)
            if ale.getRAM()[123] < obs[123]:
                lives_left -= 1
                print("Evaluation progress: {}%...                       ".format(100*(3*ep+(3-lives_left))/(3*games)), end='\r')
    log(total_reward/games, agent.games_played)
    agent.p_random_action = p_random_action
    agent.explore_unseen_states = explore_unseen_states
    

if __name__ == '__main__':
    try:
        agent = loadFromPickle("data/checkpoints/agent.pkl")
    except:
        agent = makeAgent()
        
    agent.p_random_action = makeAgent().p_random_action
    agent.explore_unseen_states = makeAgent().explore_unseen_states
    agent.getAction = makeAgent().getAction
    agent.update = makeAgent().update
    agent.trainIteration = makeAgent().trainIteration

    games_per_train = 10
    n_evaluation_games = 3
    
    games_per_evaluation = 10
    games_per_recording = 100
    
    while True:
        render_train, render_eval, render_recordings = loadBoolFromFile("data/render_bools.txt")
        train(games_per_train)
        saveToPickle("data/checkpoints/agent_{}_games.pkl".format(agent.games_played), agent)
        if agent.games_played % games_per_evaluation == 0:
            evaluate(n_evaluation_games)