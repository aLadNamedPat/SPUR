import torch
# from model import qValuePredictor
from model import qValuePredictor
from buffer import ReplayBuffer
import random
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import wandb
import torchvision.utils as vutils
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

wandb.init(
    # set the wandb project where this run will be logged
    project="RL-Modeling",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "Encoder-Decoder",
    "batch_size" : 32,
    }
)

class R_Learning:

    def __init__(
        self,
        gridSize : int,
        env,
        input_channels : int = 4,
        beta : int = 1,
        alpha : float = .005,
        tau : int = 500,
        learning_start : int = 50,
        exploration_fraction : float = 0.2,
        exploration_initial : float = 0.6,
        exploration_final : float = 0.1,
        episode_length : int = 50,
        reset_steps : int = 1800,
        buffer_size : int = 1000000,
        sample_batch_size : int = 32,
        gradient_steps : int = 1,
        lr : float = 0.0001,
        train_freq : int = 1,
    ) -> None:

        self.env = env
        self.gridSize = gridSize
        self.reset_steps = reset_steps
        # self.writer = SummaryWriter()
        self.actor = qValuePredictor(input_channels, 1, [16, 16, 32]).to(device)
        self.target_actor = qValuePredictor(input_channels ,1, [16, 16, 32]).to(device)
        # encoder_and_decoder_params = list(self.actor.encoder.parameters()) + \
        #                             list(self.actor.encoder_decoder_linear.parameters()) + \
        #                             list(self.actor.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr = lr)

        # self.optimizer2 = torch.optim.Adam(self.actor.decoder2.parameters(), lr = lr)
        self.replayBuffer = ReplayBuffer(buffer_size, sample_batch_size)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.__initalize_hyperameters__()

        self.alpha = alpha                                      # Learning rate for p
        self.beta = beta                                        # Term that prevents p from growing excessively large
        self.tau = tau                                          # The number of timesteps before updates occur
        self.learning_start = learning_start                    # The timestep at which learning begins
        self.exploration_fraction = exploration_fraction        # Percentage of training steps that will include exploration
        self.exploration_initial = exploration_initial          # Initial amount of exploration
        self.exploration_final = exploration_final              # End amount of exploration
        self.episode_length = episode_length                    # Length of each episode
        self.gradient_steps = gradient_steps                    # Number of gradient steps to take per training
        self.train_freq = train_freq                            # Number of steps until training

    def __initalize_hyperameters__(
        self
    ) -> None:
        self.reward_eps = []
        self.p = 0
        self.current_step = 0
        self.current_time = 0
        self.total_time = 0
        self.rollout_step = 0
        self.initiate = True

    def learn(
        self,
        training_steps = 100000
    ) -> None:
        self.training_steps = training_steps

        while( self.current_step < self.training_steps ):
            # print("X Steps in")
            self.rollout()

            if self.current_step > self.learning_start:
                self.train()
        self.reward_graph()

    def rollout(
        self,
        count_method : str = "decision",
    ) -> None:
        
        if self.initiate:
            obs, info = self.env.reset()
            obs = self.filter_dict(obs)
            self._last_obs = list(obs[0].values())
            self.initiate = False
            self.total_ep_reward = 0
            self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)

        self.actor.train(False)
        num_collected_steps = 0

        if count_method != "decision" and count_method != "timesteps":
            return NameError("Need to specify a valid step counting method.")
        
        while self.should_collect_more(self.train_freq, num_collected_steps):
            # print(self.current_time)
            self.current_step += 1
            num_collected_steps += 1
            self.rollout_step += 1
            action = self.sample_action() #Gets the action stores from the last observed action
            action = self.action_to_env(action)
            new_obs, rewards, dones, _, info = self.env.step(action)
            new_obs, action, rewards = self.filter_dict(new_obs, action, rewards)
            wandb.log({"Reward" : rewards[0]})

            self.replayBuffer.add(self._last_obs, action[0], rewards[0], list(new_obs[0].values()))
            self._last_obs = list(new_obs[0].values())
            self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)
            self.total_ep_reward += info["events_detected"]
            self.current_time += info['num_timesteps']
            self.total_time += info['num_timesteps']
            
            wandb.log({"Events detected" : info["events_detected"]})
            wandb.log({"Total timesteps" : self.current_time})

            # if count_method == "decision" and self.rollout_step > self.episode_length:
            #     obs, info = self.env.reset()
            #     obs = self.filter_dict(obs)
            #     self._last_obs = list(obs[0].values())
            #     self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)
            #     self.reward_eps.append(self.total_ep_reward)
            #     wandb.log({"detections per timestep" : self.total_ep_reward / self.current_time})                
            #     self.current_time = 0
            #     self.total_ep_reward = 0
            #     self.rollout_step = 0

            if count_method == "decision" and self.rollout_step > self.episode_length and self.current_step < self.reset_steps:
                obs, info = self.env.reset_agent_positions()
                obs = self.filter_dict(obs)
                self._last_obs = list(obs[0].values())
                self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)
                wandb.log({"detections per timestep" : self.total_ep_reward / self.current_time})                
                self.rollout_step = 0
           
            if count_method == "decision" and self.rollout_step > self.episode_length and self.current_step > self.reset_steps:
                wandb.log({"detections per timestep" : self.total_ep_reward / self.current_time})                

            elif count_method == "timesteps" and self.current_time > self.episode_length:
                obs, info = self.env.reset()
                obs = self.filter_dict(obs)
                self._last_obs = list(obs[0].values())
                self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)
                self.reward_eps.append(self.total_ep_reward)
                # print("Total episode reward: ", self.total_ep_reward / self.current_time)
                self.current_time = 0
                self.total_ep_reward = 0

            # print("Number of collected steps:", self.current_step)
            self.update_exploration_rate()

    # Implementation of R-Learning
    def train(
        self,
    ) -> None:
        self.actor.train(True)

        for i in range(self.gradient_steps):
            obs, actions, rewards, next_obs = self.replayBuffer.sample()

            with torch.no_grad():
                # Issue is that the result is non-flattened
                q_max = self.target_actor.find_Q_value(next_obs, (self.actor.forward(next_obs)[1] // self.gridSize, self.actor.forward(next_obs)[1] % self.gridSize))
                y = rewards.flatten() - self.p + q_max
            
            actions = (actions[:, 0], actions[:, 1])


            # self.optimizer2.zero_grad()
            # reconstruction = self.actor.reconstruction(obs)
            # loss = self.actor.reconstruction_loss(reconstruction.squeeze(1), obs[:,0])
            # loss.backward()
            # self.optimizer2.step()
            # torch.cuda.empty_cache()

            # Log the first observation and reconstruction
            # first_obs = vutils.make_grid(obs[0, 0].unsqueeze(0), normalize=True, scale_each=True)
            # first_recon = vutils.make_grid(reconstruction[0].unsqueeze(0), normalize=True, scale_each=True)

            # if self.current_step % 100 == 0:
            #     wandb.log({
            #         "First Observation": wandb.Image(first_obs.cpu()),
            #         "First Reconstruction": wandb.Image(first_recon.cpu())
            #     })
            self.optimizer.zero_grad()
            l = self.actor.find_loss(self.actor.find_Q_value(obs, actions), y)
            l.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                delta = y - self.actor.find_Q_value(obs, actions)
                condition = torch.abs(self.actor.find_Q_value(obs, actions) - self.actor.forward(obs)[0]) < self.beta

                if condition.any():
                    update_val = delta[condition].mean()

                    update_val = (rewards.flatten() - self.p).mean()
                    self.p = self.p + update_val * self.alpha
                    a =  y.mean().item()
                    b = (q_max - self.actor.find_Q_value(obs, actions)).mean().item()
                    wandb.log({"rewards sampled" : rewards.flatten().mean().item()})
                    wandb.log({"q_max" : q_max.mean().item()})
                    wandb.log({"y value" : a})
                    wandb.log({"q max - q value" : b})
                    wandb.log({"p value" : self.p})
                wandb.log({"Number of steps" : self.current_step})

            if self.current_step % 10000 == 0:
                self.replayBuffer.save_replay_buffer("replay_buffer.pkl.gz")
                
        if self.current_step % self.tau == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())

    def sample_action(
        self,
    ):
        if self.current_step < self.learning_start:
            action_ = [(self.agent_positions[0][0], self.agent_positions[0][1])]
            # print(action_)
            while ( action_[0][0] == self.agent_positions[0][0] and action_[0][1] == self.agent_positions[0][1] ):
                action_ = [(int(random.random() * self.gridSize), int(random.random() * self.gridSize))]
        else:
            self._last_obs = np.array(self._last_obs)
            with torch.no_grad():
                last_obs = torch.tensor(self._last_obs, dtype = torch.float).unsqueeze(0).to(device)
            action_ = self.predict(last_obs)
        return action_

    def predict(
        self,
        last_observation : torch.Tensor,
    ) -> torch.Tensor:
        
        if random.random() < self.exploration_rate:
            # print(self.total_time)
            # print(self.training_steps)
            # print(self.exploration_rate)
            action = [(self.agent_positions[0][0], self.agent_positions[0][1])]
            while ( action[0][0] == self.agent_positions[0][0] and action[0][1] == self.agent_positions[0][1] ):
                action = [(int(random.random() * self.gridSize), int(random.random() * self.gridSize))]
        else:
            with torch.no_grad():
                action = [(int(self.actor.choose_travel(last_observation)[1] / self.gridSize), int(self.actor.choose_travel(last_observation)[1] % self.gridSize))]
            # print(self.agent_positions[0])

        return action

    def should_collect_more(
        self,
        freq,
        num_collected_steps,
    ) -> bool:
        return freq > num_collected_steps
        
    def update_exploration_rate(
        self,
    ) -> None:
        if self.total_time / self.training_steps > self.exploration_fraction:
            self.exploration_rate = self.exploration_final
        else:
            self.exploration_rate = self.exploration_initial + (self.total_time / self.training_steps) * (self.exploration_final - self.exploration_initial) / (self.exploration_fraction)
    
    def action_to_env(
        self,
        actions : list[tuple[int, int]],
    ) -> dict:
        new_actions = {}

        for i in range(len(actions)):
            new_actions[f"agent{i}"] = actions[i]

        return new_actions

    def env_to_actions(
        self,
        actions : dict,
    ) -> list[tuple[int,int]]:
        new_actions = []

        for i in range(len(actions.values())):
            new_actions.append(actions.values()[i])

        return new_actions
    
    def filter_dict(
        self,
        obs : dict,
        actions : dict = None,
        rewards : dict = None,
    ) -> tuple:
        new_obs = []
        if actions is not None:
            new_actions = []
            new_rewards = []

        for i in range(len(obs.values())):
            new_obs.append(obs[f"agent{i}"]["observation"])
            if actions is not None:
                new_actions.append(actions[f"agent{i}"])
                new_rewards.append(rewards[f"agent{i}"])

        if actions is None:
            return new_obs
        else:
            return new_obs, new_actions, new_rewards
        

    def reward_graph(
        self,
    ) -> None:
        plt.plot(self.reward_eps, color = "red")
        plt.show()