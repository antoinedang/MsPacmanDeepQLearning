import gym
import warnings
from utils import *
from agent import *
import random
import keyboard

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    try:
        agent = loadFromPickle("data/agent.pkl")
    except:
        agent = makeAgent()
    
    original_param_groups = agent.optimizer.param_groups
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    while True:
        agent.optimizer.param_groups = original_param_groups
        saveToPickle("data/agent.pkl", agent)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = 0.1
        env.reset(seed=random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        score = 0
        while True:
            cv2.imshow('MSPACMAN',scaleImage(env.render()))
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
            _, real_reward, done, _, _ = env.step(action)
            score += real_reward
            next_obs = env.unwrapped.ale.getRAM()
            state = buildStateFromRAM(next_obs)
            reward = reward_fn(obs, next_obs, real_reward)
            agent.update(state, action, reward)
            obs = next_obs
            if done:
                print(score)
                break