import torch
from model import qValuePredictor
from buffer import ReplayBuffer
import random

class R_Learning:

    def __init__(
        self,
        gridSize : int,
        env,
        beta : int = 10,
        alpha : float = .5,
        tau : int = 10,
        learning_start : int = 100,
        exploration_fraction : float = 0.2,
        exploration_initial : float = 1.0,
        exploration_final : float = 0.02,
        episode_length : int = 1000,
        buffer_size : int = 100000,
        sample_batch_size : int = 32,
        gradient_steps : int = 1,
        lr : float = 0.0001,
        train_freq : int = 1,
    ) -> None:
        
        self.env = env
        self.actor = qValuePredictor(4, 1, [32, 16, 16])
        self.target_actor = qValuePredictor(4 ,1, [32, 16, 16])

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr = lr)

        self.replayBuffer = ReplayBuffer(buffer_size, sample_batch_size)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.__initalize_hyperameters__()

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.learning_start = learning_start
        self.exploration_fraction = exploration_fraction        # Percentage of training steps that will include exploration
        self.exploration_initial = exploration_initial          # Initial amount of exploration
        self.exploration_final = exploration_final              # End amount of exploration
        self.episode_length = episode_length                    # Length of each episode
        self.gradient_steps = gradient_steps                    # Number of gradient steps to take per training
        self.train_freq = train_freq                            # Number of steps until training

    def __initalize_hyperameters__(
        self
    ) -> None:
        self.p = 0
        self.current_step = 0
        self.current_time = 0
        self.initiate = True

    def learn(
        self,
        training_steps = 1000000
    ) -> None:
        self.training_steps = training_steps

        while( self.current_step < self.training_steps ):
            self.rollout()

            if self.current_step > self.learning_start:
                self.train()

    def rollout(
        self,
    ) -> None:
        if self.initiate:
            obs, info = self.env.reset()
            self._last_obs = obs
            self.initiate = False
            self.total_ep_reward = 0

        self.actor.train(False)
        num_collected_steps = 0

        while self.should_collect_more(self.train_freq, num_collected_steps):
            self.current_step += 1
            num_collected_steps += 1
            action = self.sample_action() #Gets the action stores from the last observed action
            new_obs, rewards, dones, info, _ = self.env.step(action)
            self.replayBuffer.add(self._last_obs, action, rewards, new_obs)
            self._last_obs = new_obs
            self.total_ep_reward += rewards
            self.current_time += info['num_timesteps']
            
            if self.current_time > self.episode_length:
                self._last_obs, info = self.env.reset()
                self.current_time = 0
                self.total_ep_reward = 0
            
            self.update_exploration_rate()
            
    # Implementation of R-Learning
    def train(
        self,
    ) -> None:
        self.actor.train(True)

        for i in range(len(self.gradient_steps)):
            obs, actions, rewards, next_obs = self.replayBuffer.sample()

            with torch.no_grad():
                q_max = self.target_actor.find_Q_value(next_obs, self.actor.forward(next_obs)[1]) # Issue is that the result is non-flattened
                y = rewards - self.p + q_max

            l = self.actor.find_loss(self.actor.find_Q_value(obs, actions), y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            delta = y - self.actor.find_Q_value(obs, actions)

            condition = torch.abs(delta - self.actor.forward(obs)[0]) < self.beta

            if condition.any():
                update_val = delta[condition].mean()
                self.p = self.p + update_val * self.alpha
            
        if self.current_step % self.tau == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())


    def sample_action(
        self,
    ):
        if self.current_step < self.learning_start:
            action_ = self.env.action_space.sample()
        else:
            last_obs = torch.tensor(self._last_obs, dtype = torch.float)
            action_ = self.predict(last_obs)

        return action_
    
    def predict(
        self,
        last_observation : torch.Tensor,
    ) -> torch.Tensor:
        if random.random() < self.exploration_rate:
            action = self.env.action_space.sample()
        else:
            action = self.actor(last_observation)[1]
        
        return action

    def should_collect_more(
        self,
        freq,
        num_collected_steps,
    ) -> bool:
        return freq > num_collected_steps
        
    def update_exploration_rate(
        self
    ):
        if self.current_step / self.training_steps > self.exploration_fraction:
            self.exploration_rate = self.exploration_final
        else:
            self.exploration_rate = self.exploration_initial + (self.current_step / self.training_steps) * (self.exploration_final - self.exploration_initial) / (self.exploration_fraction)