from agent import *
import warnings
import cv2
from utils import *
from ale_py import ALEInterface, SDL_SUPPORT

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    try:
        agent = loadFromPickle("data/checkpoints/best_agent.pkl")
    except:
        agent = makeAgent()
    
    agent.p_random_action = 0.0
    agent.explore_unseen_states = False
    agent.getAction = makeAgent().getAction
    agent.update = makeAgent().update
    agent.trainIteration = makeAgent().trainIteration
    
    ale = ALEInterface()

    ale.setInt("random_seed", random.randint(0, 9999))

    if SDL_SUPPORT:
        ale.setBool("sound", False)
        ale.setBool("display_screen", True)

    ale.loadROM("data/MSPACMAN.BIN")
    
    possible_actions = [2,3,4,5]

    seed = 0
    ale.reset_game()
    total_reward = 0
    state = None
    action = None
    
    while not ale.game_over():
        state = buildStateFromRAM(ale.getRAM(), state, action)
        action = possible_actions[state.index(max(state))]
        # action, _ = agent.getAction(state)
        reward = ale.act(action)
        total_reward += reward

    print("Total accumulated reward: {}".format(total_reward))