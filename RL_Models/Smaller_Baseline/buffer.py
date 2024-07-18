import torch
import random
from collections import deque
import numpy as np
import pickle
import gzip

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

        with torch.no_grad():
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
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float)
            action = torch.tensor(action, dtype = torch.float)
            reward = torch.tensor([reward], dtype = torch.float)
            next_state = torch.tensor(next_state, dtype = torch.float)

        self.memory.append((state, action, reward, next_state))

        if self.num_stored <= len(self.memory):
            self.num_stored += 1

        if self.num_stored >= self.batch_size:
            self.sample_ready = True

    def get_size(self):
        return len(self.memory)
    
    def save_replay_buffer(self, filename):
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self.memory, f)
        print("Replay buffer saved")

    def load_replay_buffer(self, filename = "replay_buffer.pkl.gz"):
        with gzip.open(filename, 'rb') as f:
            self.memory = pickle.load(f)
        print(f'Replay buffer loaded from {filename}')

    def split_training_testing(self, split_ratio):
        split_idx = int(len(self.memory) * split_ratio)
        training_set = list(self.memory)[:split_idx]
        testing_set = list(self.memory)[split_idx:]
        
        self.training_set = deque(training_set, maxlen=self.memory.maxlen)
        self.testing_set = deque(testing_set, maxlen=self.memory.maxlen)


    # def encoder_decoder_sampling(self, batch_size):
    #     sampled_training = random.sample(self.training_set, batch_size)
    #     sampled_testing = random.sample(self.testing_set, batch_size)

    #     with torch.no_grad():
    #         states = torch.stack([exp[0] for exp in sampled_training]).to(device)
    #         states_testing = torch.stack([exp[0] for exp in sampled_testing]).to(device)

    #     return states, states_testing
