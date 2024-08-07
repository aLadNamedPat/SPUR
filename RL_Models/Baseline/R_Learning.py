import torch
from model import qValuePredictor
from buffer import ReplayBuffer
import random
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import wandb

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
        beta : int = 10,
        alpha : float = .01,
        tau : int = 100,
        learning_start : int = 50,
        exploration_fraction : float = 0.4,
        exploration_initial : float = 0.8,
        exploration_final : float = 0.1,
        episode_length : int = 1000,
        buffer_size : int = 100000,
        sample_batch_size : int = 32,
        gradient_steps : int = 1,
        lr : float = 0.0001,
        train_freq : int = 1,
    ) -> None:

        self.env = env
        self.gridSize = gridSize
        # self.writer = SummaryWriter()
        self.actor = qValuePredictor(input_channels, 1, [32, 16, 16]).to(device)
        self.target_actor = qValuePredictor(input_channels ,1, [32, 16, 16]).to(device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr = lr)

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
            action = self.sample_action() #Gets the action stores from the last observed action'
            action = self.action_to_env(action)
            new_obs, rewards, dones, _, info = self.env.step(action)
            new_obs, action, rewards = self.filter_dict(new_obs, action, rewards)
            self.replayBuffer.add(self._last_obs, action[0], rewards[0], list(new_obs[0].values()))
            self._last_obs = list(new_obs[0].values())
            self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)
            self.total_ep_reward += info["events_detected"]
            self.current_time += info['num_timesteps']
            self.total_time += info['num_timesteps']

            if count_method == "decision" and self.rollout_step > self.episode_length:
                obs, info = self.env.reset()
                obs = self.filter_dict(obs)
                self._last_obs = list(obs[0].values())
                self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)
                self.reward_eps.append(self.total_ep_reward)
                wandb.log({"detections per timestep" : self.total_ep_reward / self.current_time})                
                self.current_time = 0
                self.total_ep_reward = 0
                self.rollout_step = 0

            elif count_method == "timesteps" and self.current_time > self.episode_length:
                obs, info = self.env.reset()
                obs = self.filter_dict(obs)
                self.last_obs = list(obs[0].values())
                self.agent_positions = np.argwhere(np.array(self._last_obs[2]) == 1)
                self.reward_eps.append(self.total_ep_reward)
                print("Total episode reward: ", self.total_ep_reward / self.current_time)
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
            
            l = self.actor.find_loss(self.actor.find_Q_value(obs, actions), y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                delta = y - self.actor.find_Q_value(obs, actions)
                condition = torch.abs(self.actor.find_Q_value(obs, actions) - self.actor.forward(obs)[0]) < self.beta

                if condition.any():
                    update_val = delta[condition].mean()
                    self.p = self.p + update_val * self.alpha
                    wandb.log({"p value" : self.p})
                    
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
            # print("here")
            with torch.no_grad():
                action = [(int(self.actor.choose_travel(last_observation)[1] // self.gridSize), int(self.actor.choose_travel(last_observation)[1] % self.gridSize))]

            # print(self.agent_positions[0])
            # print(action)

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