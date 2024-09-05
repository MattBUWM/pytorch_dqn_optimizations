import os.path
import random

import cv2
from gymnasium import Env, spaces
from pyboy import PyBoy

import numpy as np


class RewardCalculator:
    def __init__(self, pyboy: PyBoy):
        self.previous_score_low = pyboy.memory[0xc186]
        self.previous_score_high = pyboy.memory[0xc187]
        self.previous_lives = pyboy.memory[0xc211]
        self.ship_state = pyboy.memory[0xc20c]
        self.visited_positions = []

    def print_values(self, pyboy: PyBoy):
        print(self.previous_score_low, pyboy.memory[0xc186])
        print(self.previous_score_high, pyboy.memory[0xc187])
        print(self.previous_lives, pyboy.memory[0xc211])

    def get_reward(self, pyboy: PyBoy):
        new_score_low = pyboy.memory[0xc186]
        new_score_high = pyboy.memory[0xc187]
        new_lives = pyboy.memory[0xc211]
        new_ship_state = pyboy.memory[0xc20c]
        reward_value = (new_score_low - self.previous_score_low) + 255 * (new_score_high - self.previous_score_high) + 10 * (new_lives - self.previous_lives)
        if (pyboy.memory[0xc200], pyboy.memory[0xc202]) not in self.visited_positions:
            self.visited_positions.append((pyboy.memory[0xc200], pyboy.memory[0xc202]))
            reward_value += 1
        if self.ship_state < new_ship_state and new_ship_state in (2, 3, 4):
            reward_value += 10
        self.previous_score_low = new_score_low
        self.previous_score_high = new_score_high
        self.previous_lives = new_lives
        self.ship_state = new_ship_state
        return reward_value/10

    def reset(self, pyboy: PyBoy):
        self.previous_score_low = pyboy.memory[0xc186]
        self.previous_score_high = pyboy.memory[0xc187]
        self.previous_lives = pyboy.memory[0xc211]
        self.paused = pyboy.memory[0xc1f4] == 200
        self.ship_state = pyboy.memory[0xc20c]
        self.visited_positions.clear()


class ImageProcessor:
    def __init__(self, pyboy: PyBoy, flatten=False):
        self.size_in = pyboy.screen.ndarray.shape
        self.size_out = (self.size_in[0], self.size_in[1])
        self.flatten = flatten
        self.images = [np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8),
                       np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8),
                       np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8),
                       np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8)]
        if flatten:
            self.size_out = 4 * self.size_in[0] * self.size_in[1]
        else:
            self.size_out = (4, int(self.size_in[0]), int(self.size_in[1]))

    def process(self, img, grayscale=True):
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if grayscale else img
        self.images.pop(0)
        self.images.append(processed)
        array = np.array(self.images)
        if self.flatten:
            return array.flatten()
        else:
            return array

    def processed_shape(self) -> int | tuple:
        return self.size_out

    def reset(self, pyboy: PyBoy):
        self.images = [np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8),
                       np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8),
                       np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8),
                       np.zeros((self.size_in[0], self.size_in[1]), dtype=np.uint8)]


class SuperJetPakEnv(Env):
    def __init__(self,
                 game,
                 pyboy_state,
                 ticks_per_action=1,
                 force_gbc=True,
                 force_discrete=False,
                 flatten=True,
                 grayscale=False
                 ):
        super().__init__()
        if not isinstance(ticks_per_action, int):
            raise TypeError("ticks_per_action must be an integer")
        elif ticks_per_action < 1:
            raise ValueError("ticks_per_action must be a positive value")
        self.ticks_per_action = ticks_per_action
        self.pyboy = PyBoy(game, cgb=force_gbc, window='null')
        self.pyboy.set_emulation_speed(0)
        if os.path.isdir(pyboy_state):
            self.pyboy_state = os.listdir(pyboy_state)
            for x in range(len(self.pyboy_state)):
                self.pyboy_state[x] = pyboy_state + "/" + self.pyboy_state[x]
            print('list of states: ', self.pyboy_state)
            self.multi_states = True
            self.load_first = True
            self.last_state = None
        else:
            self.pyboy_state = pyboy_state
            self.multi_states = False
        if self.multi_states:
            with open(self.pyboy_state[0], "rb") as x:
                self.pyboy.load_state(x)
            print('state loaded from file ', self.pyboy_state[0])
        else:
            with open(self.pyboy_state, "rb") as x:
                self.pyboy.load_state(x)
        self.screen = self.pyboy.screen
        self._action_to_input = [
            'no_op',
            'up',
            'down',
            'left',
            'right',
            'a',
            'b',
        ]
        self.force_discrete = force_discrete
        self.flatten = flatten
        self.grayscale = grayscale
        if self.force_discrete:
            self.action_space = spaces.Discrete(len(self._action_to_input), start=0)
        else:
            self.action_space = spaces.MultiDiscrete([3, 3, 2, 2])
        self.image_process_obj = ImageProcessor(self.pyboy, flatten=self.flatten)
        self.processed_shape = self.image_process_obj.processed_shape()
        if self.flatten:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.processed_shape,), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=self.processed_shape, dtype=np.uint8)
        self.reward_obj = RewardCalculator(self.pyboy)

    def render(self, array=False):
        game_pixels_render = self.screen.ndarray.copy()
        if not array:
            game_pixels_render = cv2.resize(game_pixels_render, (game_pixels_render.shape[0] * 4, game_pixels_render.shape[1] * 4))
            cv2.imshow('game_state', game_pixels_render)
            cv2.waitKey(16)
        else:
            return cv2.cvtColor(game_pixels_render, cv2.COLOR_BGR2GRAY)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.multi_states:
            if self.load_first:
                state = self.pyboy_state[0]
                self.load_first = False
            else:
                state = None
                while state == self.last_state or state is None:
                    state = random.choice(self.pyboy_state)
            with open(state, "rb") as x:
                self.pyboy.load_state(x)
            print('state loaded from file ', state)
            self.last_state = state
        else:
            with open(self.pyboy_state, "rb") as x:
                self.pyboy.load_state(x)
        self.reward_obj.reset(self.pyboy)
        self.image_process_obj.reset(self.pyboy)
        game_pixels_render = self.screen.ndarray.copy()
        observation = self.image_process_obj.process(game_pixels_render, grayscale=self.grayscale)
        info = {}
        return observation, info

    def step(self, action):
        if self.force_discrete:
            if action != 0:
                self.pyboy.button(self._action_to_input[action], self.ticks_per_action)
        else:
            if action[0] != 0:
                self.pyboy.button(self._action_to_input[action[0] - 1], self.ticks_per_action)
            if action[1] != 0:
                self.pyboy.button(self._action_to_input[action[0] + 1], self.ticks_per_action)
            if action[2] != 0:
                self.pyboy.button(self._action_to_input[4], self.ticks_per_action)
            if action[3] != 0:
                self.pyboy.button(self._action_to_input[5], self.ticks_per_action)
        self.pyboy.tick(self.ticks_per_action)
        game_pixels_render = self.screen.ndarray.copy()
        observation = self.image_process_obj.process(game_pixels_render, grayscale=self.grayscale)
        reward = self.reward_obj.get_reward(self.pyboy)
        info = {}
        truncated = False
        terminated = self.pyboy.memory[0xc248] != 1
        info['level'] = self.pyboy.memory[0xc248]

        return observation, reward, terminated, truncated, info

    def close(self):
        self.pyboy.stop(save=False)
