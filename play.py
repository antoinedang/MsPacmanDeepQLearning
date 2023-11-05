import gym
from agent import RLAgent
import warnings
import cv2
from utils import *

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    
    agent = loadFromPickle("data/agent.pkl")
    agent.p_random_action = 0.0
    agent.explore_unseen_states = False

    env.reset()
    total_reward = 0
    state = None
    
    while True:
        cv2.imshow('',scaleImage(env.render()))
        cv2.waitKey(1)
        
        state = buildStateFromRAM(env.unwrapped.ale.getRAM())
        action, _ = agent.getAction(state)
        _, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if done:
            break

    print("Total accumulated reward: {}".format(total_reward))
    env.close()