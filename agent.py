from itertools import permutations
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils import *

ghost_coordinates_combos = list(permutations([0,1,2,3]))

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
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model =  nn.Sequential(*layers)  
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        self.input_size = input_size
        self.games_played = 0
        self.observed_state_actions = {}
        
        self.p_random_action = p_random_action
        
        self.explore_unseen_states = explore_unseen_states
        self.threshold_close_state = 10
        self.max_cached_states_per_action = 100
        self.time_between_state_caching = 30
        self.time_since_last_state_cache = self.time_between_state_caching
        
    def getAction(self, state):
        #RANDOM ACTION
        if random.random() < self.p_random_action:
            return random.randint(1,4), 30
        #NORMAL ACTION
        action_rewards = [-99999999,0,0,0,0]
        for action in [1,2,3,4]:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_tensor = torch.tensor([0,0,0,0], dtype=torch.float32)
            action_tensor[action-1] = 1.0
            input_tensor = torch.cat((state_tensor, action_tensor), dim=0).to(self.device)
            expected_reward = float(self.model(input_tensor).cpu()[0])
            #WEIGHT ACTION REWARDS BASED ON UNSEEN STATES
            if self.explore_unseen_states:
                try:
                    closest_seen_state_dist = min([stateDistance(s,state) for s in self.observed_state_actions[action]])
                    closest_seen_state_dist = 1 if closest_seen_state_dist < self.threshold_close_state else closest_seen_state_dist
                    action_rewards[action] = expected_reward * (1/closest_seen_state_dist) + 10000000 * (1.0 - (1/closest_seen_state_dist))
                except:
                    action_rewards[action] = expected_reward
            else:
                action_rewards[action] = expected_reward
        max_q_action = max(action_rewards)
        return action_rewards.index(max_q_action), 0
    
    def trainIteration(self):
        num_states_in_buffer = len(self.observed_state_actions.get(1, [])) + len(self.observed_state_actions.get(2, [])) + len(self.observed_state_actions.get(3, [])) + len(self.observed_state_actions.get(4, []))
        batch = torch.zeros((num_states_in_buffer, self.input_size))
        rewards = torch.zeros((num_states_in_buffer, 1))
        num_states_added_to_batch = 0
        for action in [1,2,3,4]:
            for state, reward in self.observed_state_actions[action]:
                state_tensor = torch.tensor(state)
                action_tensor = torch.tensor([0,0,0,0])
                action_tensor[action-1] = 1.0
                input_tensor = torch.cat((state_tensor, action_tensor), dim=0)
                batch[num_states_added_to_batch] = input_tensor
                rewards[num_states_added_to_batch] = reward
                num_states_added_to_batch += 1

        batch = batch.to(self.device)
        rewards = rewards.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = self.criterion(outputs, rewards)
        loss.backward()
        self.optimizer.step()
                
    
    def update(self, state, action, reward):        
        ## NORMALIZE STATE AND ACTION DATA
        state_copy = copy.deepcopy(state)
        ## ADD ALL COMBINATIONS OF SYMMETRY
        for _ in range(2):
            state_copy[0] *= -1
            state[0] *= -1
            state_copy[1] *= -1
            state[1] *= -1
            state_copy[2] *= -1
            state[2] *= -1
            state_copy[3] *= -1
            state[3] *= -1
            state_copy[8] = 176 - state_copy[8]
            state[8] = 176 - state[8]
            
            if action == 2: action = 3
            elif action == 3: action = 2
            
            action_tensor = torch.tensor([0,0,0,0])
            action_tensor[action-1] = 1.0
            
            for ghost_coordinates_combo in ghost_coordinates_combos:
                for i in ghost_coordinates_combo:
                    state_copy[i] = state[i]
                    state_copy[i+4] = state[i+4]
                ## CACHE SEEN STATES
                if self.time_since_last_state_cache < self.time_between_state_caching:
                    self.time_since_last_state_cache += 1
                self.time_since_last_state_cache = 0
                try:
                    self.observed_state_actions[action].append((state_copy, reward))
                    if len(self.observed_state_actions[action]) > self.max_cached_states:
                        self.observed_state_actions[action].remove(0)
                except:
                    self.observed_state_actions[action] = []
                    self.observed_state_actions[action].append((state_copy, reward))
        
        self.train()
    

def makeAgent():
    return RLAgent(p_random_action=0.05, input_size=14, hidden_sizes=[128, 128], explore_unseen_states=True)