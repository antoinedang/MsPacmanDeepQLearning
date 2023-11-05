import gym
from agent import RLAgent
import warnings
from utils import *
import random

# Suppress all warnings
warnings.filterwarnings("ignore")

def log(score, games_played):
    # log to CSV file
    appendToFile("{},{}".format(score, games_played), "data/" + run_name + ".score_per_games_played.csv")
    pass

def train(episodes):
    env = gym.make("ALE/MsPacman-v5", render_mode='human', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    
    total_reward = 0
    for ep in range(episodes):
        env.reset(seed=random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        while True:
            env.render()
            action = agent.getAction(state)
            _, real_reward, done, _, _ = env.step(action)
            total_reward += real_reward
            next_obs = env.unwrapped.ale.getRAM()
            state = buildStateFromRAM(next_obs)
            reward = reward_fn(obs, next_obs, real_reward)
            agent.update(state, action, reward)
            obs = next_obs
            if done:
                break
        agent.games_played += 1
        print("Epoch progress: {}%...           ".format(100*(ep+1)/episodes), end='\r')
        
    log(total_reward/episodes, agent.games_played)
    env.close()
    
def generateGameplayVideos(num_games):
    env = gym.make("ALE/MsPacman-v5", render_mode='human', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    # env = gym.wrappers.RecordVideo(env, 'recordings', episode_trigger = lambda x: True)
    for i in range(num_games):
        env.reset(seed=random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        total_reward = 0
        while True:
            env.render()
            action = agent.getAction(state)
            _, real_reward, done, _, _ = env.step(action)
            state = buildStateFromRAM(env.unwrapped.ale.getRAM())
            total_reward += reward_fn(obs, env.unwrapped.ale.getRAM(), real_reward)
            obs = env.unwrapped.ale.getRAM()
            if done:
                break
    env.close()

if __name__ == '__main__':
    try:
        agent = loadFromPickle("data/agent.pkl")
    except:
        agent = RLAgent(p_random_action=0.0, input_size=11, hidden_sizes=[128, 8], explore_unseen_states=True)
    
    run_name = "simple_dist_function"

    games_per_epoch = 5
    num_recordings_per_epoch = 1
    
    while True:
        train(games_per_epoch)
        
        saveToPickle("data/agent.pkl", agent)
        
        generateGameplayVideos(num_recordings_per_epoch)