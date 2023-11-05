import np
import torch

class RLAgent:
    def __init__(self, p_random_action, checkpoint_file):
        self.games_played = 0
        self.observed_state_actions = []
        self.p_random_action = p_random_action
        # exploration via random actions or through optimistic priors for unseen state/action pairs
        # define how optimistic priors 
        # generalization methods: 
        pass
    def reset(self):
        self.observed_state_actions = []
        
    def getAction(self, state):
        
        # print(state)
        return 0
    def update(self, state, action, reward, next_state, done):
        pass
    def saveCheckpoint(self, filename):
        pass
