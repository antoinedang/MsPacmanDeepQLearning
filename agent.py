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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
        self.input_size = input_size
        self.p_random_action = p_random_action
        self.games_played = 0
        self.observed_state_actions = {}
        self.explore_unseen_states = explore_unseen_states
        self.threshold_close_state = 20
        self.max_cached_states = 100
        
    def getAction(self, state):
        #RANDOM ACTION
        if random.random() < self.p_random_action:
            return random.randint(1,4), 30
        #NORMAL ACTION
        action_rewards = [-99999999,0,0,0,0]
        for action in [1.0,2.0,3.0,4.0]:
            state_tensor = torch.tensor(state)
            action_tensor = torch.tensor([action])
            input_tensor = torch.cat((state_tensor, action_tensor), dim=0).to(self.device)
            expected_reward = float(self.model(input_tensor).cpu()[0])
            #WEIGHT ACTION REWARDS BASED ON UNSEEN STATES
            if self.explore_unseen_states:
                try:
                    closest_seen_state_dist = min([stateDistance(s,state) for s in self.observed_state_actions[int(action)]])
                    closest_seen_state_dist = 1 if closest_seen_state_dist < self.threshold_close_state else closest_seen_state_dist
                    action_rewards[int(action)] = expected_reward + 10000000 * (1.0 - (1/closest_seen_state_dist))
                except:
                    action_rewards[int(action)] = expected_reward
            else:
                action_rewards[int(action)] = expected_reward
        max_q_action = max(action_rewards)
        return action_rewards.index(max_q_action), 0

    def update(self, state, action, reward):
        if self.explore_unseen_states:
            try:
                self.observed_state_actions[action].append(state)
                if len(self.observed_state_actions[action]) > self.max_cached_states:
                    self.observed_state_actions[action].remove(random.randint(0,len(self.observed_state_actions[action])))
            except:
                self.observed_state_actions[action] = []
                self.observed_state_actions[action].append(state)
        
        ## ADD ALL COMBINATIONS OF SYMMETRY
        map_width = 2*88
        for i in range(2):
            state[0] = map_width - state[0]
            state[1] = map_width - state[1]
            state[2] = map_width - state[2]
            state[3] = map_width - state[3]
            state[8] = map_width - state[8]
            
            ghost_coordinates_x = copy.deepcopy(state[0:4])
            ghost_coordinates_y = copy.deepcopy(state[4:8])
            
            batch = torch.zeros((len(ghost_coordinates_combos), self.input_size))

            if action == 1: action = 4
            elif action == 2: action = 3
            elif action == 3: action = 2
            elif action == 4:  action = 1
            action_tensor = torch.tensor([float(action)])
            rewards = torch.tensor([float(reward)]*len(ghost_coordinates_combos)).to(self.device)
            
            for ghost_coordinates_combo in ghost_coordinates_combos:
                for i in ghost_coordinates_combo:
                    state[i] = ghost_coordinates_x[i]
                    state[i+4] = ghost_coordinates_y[i]
                state_tensor = torch.tensor(state)
                batch[i] = torch.cat((state_tensor, action_tensor), dim=0)
            
        batch = batch.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = self.criterion(outputs, rewards)
        loss.backward()
        self.optimizer.step()
    
