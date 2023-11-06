import gym
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
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    print("Epoch progress: 0%                  ", end='\r')
    for ep in range(episodes):
        env.reset(seed=random.randint(0, 10000))
        random.seed(random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        next_is_random = 0
        lives_left = 3
        while True:
            if render_train:
                game_img = env.render()
                cv2.imshow('MSPACMAN',scaleImage(game_img))
                cv2.waitKey(1)
            if next_is_random > 0:
                next_is_random -= 1
            else:
                action, next_is_random = agent.getAction(state)
            _, real_reward, done, _, _ = env.step(action)
            next_obs = env.unwrapped.ale.getRAM()
            state = buildStateFromRAM(next_obs)
            reward = reward_fn(obs, next_obs, real_reward)
            if next_obs[123] < obs[123]:
                lives_left -= 1
                print("Epoch progress: {}%...               ".format(100*(3*ep+(3-lives_left))/(3*episodes)), end='\r')
            agent.update(state, action, reward)
            obs = next_obs
            if done:
                break
        agent.games_played += 1
    env.close()
    
def generateGameplayVideos(num_games_played):
    p_random_action = agent.p_random_action
    explore_unseen_states = agent.explore_unseen_states
    agent.p_random_action = 0
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    print("Recording progress: 0%                  ", end='\r')
    for i in range(3):
        env.reset(seed=random.randint(0, 10000))
        random.seed(random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        video = []
        while True:
            game_img = env.render()
            video.append(game_img)
            if render_recordings:
                cv2.imshow('MSPACMAN-RECORDING',scaleImage(game_img))
                cv2.waitKey(1)
            action, _ = agent.getAction(state)
            _, _, done, _, _ = env.step(action)
            state = buildStateFromRAM(env.unwrapped.ale.getRAM())
            obs = env.unwrapped.ale.getRAM()
            if done:
                break
        print("Recording progress: {}%...               ".format((i+1)/3), end='\r')
        out = cv2.VideoWriter("recordings/{}_games_played_vid_{}.mp4".format(num_games_played, i+1), cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (game_img.shape[1], game_img.shape[0]))
        for frame in video:
            out.write(frame)
    env.close()
    agent.p_random_action = p_random_action
    agent.explore_unseen_states = explore_unseen_states

def evaluate(games):
    p_random_action = agent.p_random_action
    explore_unseen_states = agent.explore_unseen_states
    agent.p_random_action = 0
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    print("Evaluation progress: 0%                  ", end='\r')
    total_reward = 0
    for ep in range(games):
        env.reset(seed=random.randint(0, 10000))
        random.seed(random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        lives_left = 3
        while True:
            if render_eval:
                game_img = env.render()
                cv2.imshow('MSPACMAN',scaleImage(game_img))
                cv2.waitKey(1)
            action, _ = agent.getAction(state)
            _, real_reward, done, _, _ = env.step(action)
            total_reward += real_reward
            next_obs = env.unwrapped.ale.getRAM()
            state = buildStateFromRAM(next_obs)
            if next_obs[123] < obs[123]:
                lives_left -= 1
                print("Epoch progress: {}%...               ".format(100*(3*ep+(3-lives_left))/(3*games)), end='\r')
            obs = next_obs
            if done:
                break
    log(total_reward/games, agent.games_played)
    env.close()
    agent.p_random_action = p_random_action
    agent.explore_unseen_states = explore_unseen_states
    

if __name__ == '__main__':
    try:
        agent = loadFromPickle("data/checkpoints/agent.pkl")
    except:
        agent = makeAgent()

    games_per_train = 10
    n_evaluation_games = 3
    
    games_per_evaluation = 5
    games_per_recording = 100
    
    while True:
        render_train, render_eval, render_recordings = loadBoolFromFile("data/render_bools.txt")
        train(games_per_train)
        saveToPickle("data/checkpoints/agent_{}_games.pkl".format(agent.games_played), agent)
        if agent.games_played % games_per_evaluation == 0:
            evaluate(games_per_evaluation)
        if agent.games_played % games_per_recording == 0:
            generateGameplayVideos(agent.games_played)