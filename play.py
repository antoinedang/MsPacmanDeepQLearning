from agent import *
import warnings
import cv2
from utils import *

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    while True:
        agent = loadFromPickle("data/checkpoints/best_agent.pkl")
        agent.p_random_action = 0.0
        agent.explore_unseen_states = False
        seed = random.randint(0, 9999)
        ale = makeEnvironment(render=True, seed=3623)
        
        possible_actions = [2,3,4,5]

        total_reward = 0
        state = None
        action = None
        
        while not ale.game_over():
            state = buildStateFromRAM(ale.getRAM(), state, action)
            action, _ = agent.getAction(state)
            reward = ale.act(action)
            total_reward += reward

        print("Seed {} - Total accumulated reward: {}".format(seed, total_reward))
        appendToFile('{},{}'.format(seed, total_reward), 'data/reward_per_seed.csv')