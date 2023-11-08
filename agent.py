from itertools import permutations
import random
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

class RLAgent(nn.Module):
    def __init__(self, p_random_action, input_size, hidden_sizes, explore_unseen_states):
        super().__init__()
        layers = []
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], 4))
        layers.append(nn.Softmax(dim=0))
        self.model =  nn.Sequential(*layers)  
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.input_size = input_size
        self.games_played = 0
        self.observed_state_actions = {}
        
        self.p_random_action = p_random_action
        
        self.explore_unseen_states = explore_unseen_states
        self.threshold_close_state = 10
        self.max_cached_states_per_action = 1
        self.time_between_state_caching = 1
        self.random_action_repeat = 30
        self.time_since_last_state_cache = self.time_between_state_caching
        
    def getAction(self, state):
        #RANDOM ACTION
        if random.random() < self.p_random_action:
            return random.randint(1,4), self.random_action_repeat
        #NORMAL ACTION        
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        expected_rewards = self.model(state_tensor).cpu()
        #WEIGHT ACTION REWARDS BASED ON UNSEEN STATES
        if self.explore_unseen_states:
            for action in [1,2,3,4]:
                try:
                    closest_seen_state_dist = min([stateDistance(s,state) for s in self.observed_state_actions[action]])
                    closest_seen_state_dist = 1 if closest_seen_state_dist < self.threshold_close_state else closest_seen_state_dist
                    expected_rewards[action-1] = expected_rewards[action-1] * (1/closest_seen_state_dist) + 10000000 * (1.0 - (1/closest_seen_state_dist))
                except:
                    pass
        
        opt_action = (torch.argmax(expected_rewards).item())+1
            
        return opt_action, 0
    
    def trainIteration(self):
        num_states_in_buffer = len(self.observed_state_actions.get(1, [])) + len(self.observed_state_actions.get(2, [])) + len(self.observed_state_actions.get(3, [])) + len(self.observed_state_actions.get(4, []))
        batch = torch.zeros((num_states_in_buffer, self.input_size))
        rewards = []
        num_states_added_to_batch = 0
        for action in [1,2,3,4]:
            for state, reward in self.observed_state_actions.get(action, []):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                batch[num_states_added_to_batch] = state_tensor
                rewards.append((action-1, reward))
                num_states_added_to_batch += 1

        batch = batch.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        rewards_tensor = torch.clone(outputs)
        for i in range(num_states_added_to_batch): rewards_tensor[i][rewards[i][0]] = float(rewards[i][1])
        loss = self.criterion(outputs, rewards_tensor)
        # print("loss", loss)
        loss.backward()
        self.optimizer.step()
                
    def update(self, state, action, reward):        
        ## NORMALIZE STATE AND ACTION DATA
        if self.time_since_last_state_cache < self.time_between_state_caching and abs(reward) < 0.5: # don't skip caches on very heavily penalized states
            self.time_since_last_state_cache += 1
            self.trainIteration()
            return
        self.time_since_last_state_cache = 0
        ## CACHE SEEN STATES
        try:
            self.observed_state_actions[action].append((state, reward))
            if len(self.observed_state_actions[action]) > self.max_cached_states_per_action:
                # self.observed_state_actions[action].pop(random.randint(0,len(self.observed_state_actions[action])-16))
                self.observed_state_actions[action].pop(0)
        except:
            self.observed_state_actions[action] = []
            self.observed_state_actions[action].append((state, reward))
        
        self.trainIteration()

def makeAgent():
    return RLAgent(p_random_action=2.0/60, input_size=4, hidden_sizes=[8, 8], explore_unseen_states=True)