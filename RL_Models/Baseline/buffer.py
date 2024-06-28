import torch
import random
from collections import deque
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    #I need to store the past rewards, states, actions, and 
    def __init__(self, max_len = 1e6, batch_size = 32):
        self.memory = deque(maxlen=max_len)
        self.num_stored = 0
        self.sample_ready = False
        self.batch_size = batch_size
        
    def sample(self):
        sampled_experiences = random.sample(self.memory, self.batch_size)

        states = torch.stack([exp[0] for exp in sampled_experiences]).to(device)
        actions = torch.stack([exp[1] for exp in sampled_experiences]).to(device)
        rewards = torch.stack([exp[2] for exp in sampled_experiences]).to(device)
        next_states = torch.stack([exp[3] for exp in sampled_experiences]).to(device)

        return states, actions, rewards, next_states

    def add(self, state, action, reward, next_state):

        # print(state)
        # print(action)
        # print(reward)

        state = np.array(state)
        next_state = np.array(next_state)
        state = torch.tensor(state, dtype = torch.float).to(device)
        action = torch.tensor(action, dtype = torch.float).to(device)
        reward = torch.tensor([reward], dtype = torch.float).to(device)
        next_state = torch.tensor(next_state, dtype = torch.float).to(device)

        self.memory.append((state, action, reward, next_state))

        if self.num_stored <= len(self.memory):
            self.num_stored += 1

        if self.num_stored >= self.batch_size:
            self.sample_ready = True

    def get_size(self):
        return len(self.memory)