from agent import *
import warnings
import cv2
from utils import *

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env = makeEnvironment()
    
    try:
        agent = loadFromPickle("data/checkpoints/best_agent.pkl")
    except:
        agent = makeAgent()
    
    agent.p_random_action = 0.0
    agent.explore_unseen_states = False
    agent.getAction = makeAgent().getAction
    agent.update = makeAgent().update
    agent.trainIteration = makeAgent().trainIteration

    seed = 0
    env.reset(seed=seed)
    random.seed(seed)
    total_reward = 0
    state = None
    
    while True:
        cv2.imshow('',scaleImage(env.render()))
        cv2.waitKey(1)
        
        
        state = buildStateFromRAM(env.unwrapped.ale.getRAM(), state)
        action = state.index(max(state))+1
        # action, _ = agent.getAction(state)
        _, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if done:
            break

    print("Total accumulated reward: {}".format(total_reward))
    env.close()