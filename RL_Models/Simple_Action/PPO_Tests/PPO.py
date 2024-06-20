import torch
from torch.nn import functional as F
from torch.distributions import Normal, Categorical # Assuming continuous actions; adjust if discrete
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("../"))
from model import ActorCNN, CriticCNN


class PPO():
    def __init__(self, env):
        self.env = env
        self.__initialize__hyperparams()
        a = env.agents
        self.action_dim = 4
        self.obs_dim = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = ActorCNN(self.obs_dim, self.action_dim).to(self.device)
        self.critic = CriticCNN(self.obs_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic) 
        self.global_step = 0

    def batch_train(self, total_time_steps):
        self.actor.train()
        self.critic.train()
        t = 0

        while t < total_time_steps:
            batch_obs, batch_acts, batch_log_acts, batch_rtgs, batch_lens = self.batch_updates() #Obtain the batch updates
            adv = batch_rtgs - self.critic(batch_obs).squeeze().detach() #Compute the advantage
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) #Normalize the advantage
            for i in range(self.num_epochs_per_training):
                action_prob = self.actor(batch_obs) #This is the discretized action probability that is generated
                dist = Categorical(action_prob)
                curr_log_prob = dist.log_prob(batch_acts) #Compute the current log probability from the batch_actions that were sampled
                p_weight = (curr_log_prob - batch_log_acts).exp()
                clipped_weight = torch.clamp(p_weight, 1 - self.clip, 1 + self.clip)
                comp1 = p_weight * adv
                comp2 = clipped_weight * adv 
                actor_loss = (-torch.min(comp1, comp2)).mean()
                
                Val = self.critic(batch_obs).squeeze()
                critic_loss = F.smooth_l1_loss(batch_rtgs, Val)
                actor_loss = actor_loss.requires_grad_(True)
                critic_loss = critic_loss.requires_grad_(True) 

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

            t += np.sum(batch_lens)
            print(f"total_timesteps: {t}")

    def __initialize__hyperparams(self):
        self.timesteps_per_episode = 256
        self.batch_timesteps = 2048
        self.num_epochs_per_training = 5
        self.gamma = 0.99
        self.clip = 0.3
        self.lr_actor = 0.0003
        self.lr_critic = 0.0003

    def batch_updates(self):
        batch_obs = []                      # Save the observations
        batch_rews = []                     # Save the rewards gained per episode
        batch_lens = []                     # Save the length of the episodes
        batch_acts = []                     # Save the actions taken
        batch_log_acts = []                 # Compute the log probabilities of the actions taken
        batch_rtgs = []                     # Compute the batch returns to go
        t = 0
        cumulative_rewards = 0
        while t < self.batch_timesteps:
            episode_reward = []
            obs, info = self.env.reset()
            obs = obs["agent0"]["observation"]
            eRewards = obs['eReward']
            pGrid = obs["pGrid"]
            agent_location = obs["agent"]
            obs = torch.tensor(np.stack((eRewards, pGrid, agent_location)), dtype=torch.float).to(self.device)
            done = False
            for ep_len in range(self.timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, action_log_prob = self.take_action(obs.unsqueeze(0))
                action = {"agent0" : action}
                obs, reward, done, truncated, info = self.env.step(action)
                if truncated["agent0"]:
                    break
                obs = obs["agent0"]["observation"]
                eRewards = obs['eReward']
                pGrid = obs["pGrid"]
                agent_location = obs["agent"]
                obs = torch.tensor(np.stack((eRewards, pGrid, agent_location)), dtype=torch.float).to(self.device)

                cumulative_rewards += reward["agent0"]
                batch_acts.append(action["agent0"])
                episode_reward.append(reward["agent0"])
                batch_log_acts.append(action_log_prob)
            batch_lens.append(ep_len + 1)
            batch_rews.append(episode_reward)
            print(cumulative_rewards)
        self.global_step += 1

        batch_obs = torch.stack(batch_obs).to(self.device)
        batch_acts = torch.stack(batch_acts).to(self.device)
        batch_log_acts = torch.stack(batch_log_acts).to(self.device)
        batch_rtgs = torch.tensor(self.find_rewards_to_go(batch_rews), dtype=torch.float).to(self.device)
        return batch_obs, batch_acts, batch_log_acts, batch_rtgs, batch_lens
    

    def find_rewards_to_go(self, rewards):
        #Store the returns from each episode at every state
        rtgs = []
        for rew_episodes in reversed(rewards):
            discounted_reward = 0
            for reward in reversed(rew_episodes):
                discounted_reward = reward + self.gamma * discounted_reward
                rtgs.insert(0, discounted_reward) #Place the new discounted reward at the front of the list (since the rewards are reversed)
        return rtgs

    def take_action(self, obs):
        action = self.actor(obs)
        dist = Categorical(action)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).detach()
        return action.squeeze(), action_log_prob
    
    def evaluate_value(self, obs):
        action_prob = self.actor(obs)
        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs).squeeze()
        return value, log_prob
        
    def save_model(self, filename='ppo_model_approach.pth'):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load_model(self, filename='ppo_model_approach.pth'):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])