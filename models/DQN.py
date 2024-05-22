from __future__ import annotations

import json
import math
import os
import random
from collections import deque, namedtuple
from itertools import count

import torch
from torch import nn

import models.BaseModel as BaseModel

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(BaseModel.BaseModel):
    def __init__(self, parameters, load_existing=False, load_specific=None):
        super().__init__(parameters, load_existing)
        self.obs_shape = parameters['obs_shape']
        self.action_shape = parameters['action_shape']
        self.batch_size = parameters['batch_size']
        self.replay_memory_capacity = parameters['replay_memory_capacity']
        self.replay_mem = deque(maxlen=self.replay_memory_capacity)
        self.gamma = parameters['gamma']
        self.tau = parameters['tau']
        self.epsilon = parameters['epsilon']
        self.target_net = None
        self.policy_net = None
        self.optimizer = None
        self._init_network()
        self.statistics = {
            'episode_rewards_mean': [],
            'episode_rewards_sum': [],
            'episode_lengths': [],
            'episode_loss_mean': []
        }
        self.current_episode_loss = []
        if load_existing:
            self._load(load_specific)

    def _init_network(self):
        self.target_net = BaseModel.get_network(self.network_type, self.obs_shape, self.action_shape, self.activation_function).to(self.device)
        self.policy_net = BaseModel.get_network(self.network_type, self.obs_shape, self.action_shape, self.activation_function).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.optimizer_parameters['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.optimizer_parameters['lr'], amsgrad=self.optimizer_parameters['amsgrad'])
        if self.optimizer_parameters['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.optimizer_parameters['lr'], amsgrad=self.optimizer_parameters['amsgrad'])

    def _load(self, load_specific=None):
        if load_specific is None:
            self.target_net.load_state_dict(torch.load(self.model_path + '/target/checkpoint' + str(self.current_episode) + '.pt'))
            self.policy_net.load_state_dict(torch.load(self.model_path + '/policy/checkpoint' + str(self.current_episode) + '.pt'))
        else:
            self.target_net.load_state_dict(
                torch.load(self.model_path + '/target/checkpoint' + str(load_specific) + '.pt'))
            self.policy_net.load_state_dict(
                torch.load(self.model_path + '/policy/checkpoint' + str(load_specific) + '.pt'))
        self.optimizer.load_state_dict(torch.load(self.model_path + '/optimizer.pt'))
        self.replay_mem = torch.load(self.model_path + '/replay.temp')

    def training_episode(self, env):
        print('Starting training episode ', self.current_episode + 1)
        episode_rewards = []
        self.current_episode_loss.clear()
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        for t in count():
            action = self.predict(state, env, training=True)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_rewards.append(reward)
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.replay_mem.append(Transition(state, action, next_state, reward))

            state = next_state

            self._optimize()

            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

            if done:
                self.statistics['episode_lengths'].append(t + 1)
                rewards_sum = sum(episode_rewards)
                self.statistics['episode_rewards_sum'].append(rewards_sum)
                self.statistics['episode_rewards_mean'].append(rewards_sum / len(episode_rewards))
                self.statistics['episode_loss_mean'].append(sum(self.current_episode_loss) / len(self.current_episode_loss))
                print("episode length:", self.statistics['episode_lengths'][-1])
                print("episode reward sum:", self.statistics['episode_rewards_sum'][-1])
                print("episode mean reward :", self.statistics['episode_rewards_mean'][-1])
                print("episode mean loss:", self.statistics['episode_loss_mean'][-1])
                break

    def _optimize(self):
        if len(self.replay_mem) < self.batch_size:
            return
        transitions = random.sample(self.replay_mem, self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.current_episode_loss.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def predict(self, observation, env, training=False):
        if training:
            sample = random.random()
            threshold = self.epsilon['end'] + (self.epsilon['start'] - self.epsilon['end']) * math.exp(-1. * self.steps_done / self.epsilon['decay'])
            self.steps_done += 1
            if sample > threshold:
                with torch.no_grad():
                    return self.policy_net(observation).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                return self.target_net(observation).max(1).indices.view(1, 1)

    def save(self, full_save=True):
        if not os.path.exists(self.model_path + '/target'):
            os.makedirs(self.model_path + '/target')
        if not os.path.exists(self.model_path + '/policy'):
            os.makedirs(self.model_path + '/policy')
        if full_save:
            parameters = {
                'model_path': self.model_path,
                'optimizer_parameters': self.optimizer_parameters,
                'network': self.network_type,
                'activation_function': self.activation_function,
                'obs_shape': self.obs_shape,
                'action_shape': self.action_shape,
                'target_episode': self.target_episode,
                'current_episode': self.current_episode,
                'steps_done': self.steps_done,
                'batch_size': self.batch_size,
                'save_freq': self.save_freq,
                'replay_memory_capacity': self.replay_memory_capacity,
                'gamma': self.gamma,
                'tau': self.tau,
                'epsilon': self.epsilon
            }
            with open(self.model_path + '/parameters.json', 'w') as f:
                json.dump(parameters, f, ensure_ascii=False, indent=4)
            with open(self.model_path + '/statistics.json', 'w') as f:
                json.dump(self.statistics, f, ensure_ascii=False, indent=4)
            torch.save(self.optimizer.state_dict(), self.model_path + '/optimizer.pt')
            torch.save(self.replay_mem, self.model_path + '/replay.temp')
        torch.save(self.target_net.state_dict(),
                   self.model_path + '/target/checkpoint' + str(self.current_episode) + '.pt')
        torch.save(self.policy_net.state_dict(),
                   self.model_path + '/policy/checkpoint' + str(self.current_episode) + '.pt')

    @staticmethod
    def load(model_path, load_specific=None) -> DQN:
        parameters = json.load(open(model_path + '/parameters.json'))
        model = DQN(parameters, load_existing=True, load_specific=load_specific)
        return model
