import warnings
from utils import *
from agent import *
import random
import keyboard

# Suppress all warnings
warnings.filterwarnings("ignore")

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
    print(agent.games_played)
    
    env = makeEnvironment()
    while True:
        saveToPickle("data/checkpoints/expert_agent.pkl", agent)
        env.reset(seed=random.randint(0, 10000))
        random.seed(random.randint(0, 10000))
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
            state = buildStateFromRAM(next_obs, state, action)
            agent.update(state, action, obs)
            obs = next_obs
            if done:
                print(score)
                break