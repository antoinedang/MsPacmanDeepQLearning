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
    env = gym.make("ALE/MsPacman-v5", render_mode=None, full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    
    for ep in range(episodes):
        obs = env.reset(seed=random.randint(-1000, 1000))
        total_reward = 0
        state = None
        while True:
            state = buildStateFromRAM(obs, state)
            action = agent.getAction(state)
            next_obs, reward, done, _, _ = env.step(action)
            next_state = buildStateFromRAM(next_obs, state)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            obs = next_obs
            if done:
                break
        agent.games_played += 1
        log(total_reward, agent.games_played)
        print("Epoch progress: {}%...           ".format(100*(ep+1)/episodes), end='\r')
            
    env.close()
    
def generateGameplayVideos(num_games):
    env = gym.make("ALE/MsPacman-v5", render_mode='human', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    # env = gym.wrappers.RecordVideo(env, 'recordings', episode_trigger = lambda x: True)
    for i in range(num_games):
        obs = env.reset(seed=random.randint(-1000, 1000))
        while True:
            env.render()
            action = agent.getAction(obs)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
    env.close()
    

if __name__ == '__main__':
    agent_checkpoint = "latest.pt"
    agent = RLAgent("", agent_checkpoint)
    
    run_name = "simple_dist_function"

    games_per_epoch = 5
    num_recordings_per_epoch = 1
    
    while True:
        train(games_per_epoch)
        
        agent.saveCheckpoint(agent_checkpoint)
        
        generateGameplayVideos(num_recordings_per_epoch)
        
    
    
    
   