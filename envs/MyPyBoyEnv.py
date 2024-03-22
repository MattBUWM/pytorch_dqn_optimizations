from gymnasium import Env, spaces
from pyboy import PyBoy, WindowEvent

import numpy as np


class MyPyBoyEnv(Env):
    def __init__(self, game, pyboy_state, ticks_per_action=1, force_gbc=True):
        super().__init__()
        if not isinstance(ticks_per_action, int):
            raise TypeError("ticks_per_action must be an integer")
        elif ticks_per_action < 1:
            raise ValueError("ticks_per_action must be a positive value")
        self.ticks_per_action = ticks_per_action
        self.pyboy = PyBoy(game, cgb=force_gbc)
        self.pyboy.set_emulation_speed(4)
        self.pyboy_state = open(pyboy_state, "rb")
        self.pyboy.load_state(self.pyboy_state)
        self.screen = self.pyboy.botsupport_manager().screen()
        self._action_to_input = {
            0: (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            1: (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            2: (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            3: (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
            4: (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            5: (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
            6: (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
            7: (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
        }
        self.action_space = spaces.Discrete(len(self._action_to_input))
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)

    def render(self):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        return game_pixels_render

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pyboy.load_state(self.pyboy_state)

    def step(self, action):
        self.pyboy.send_input(self._action_to_input[action][0])
        for _ in range(self.ticks_per_action):
            self.pyboy.tick()
        self.pyboy.send_input(self._action_to_input[action][1])
        observation = self.render()
        reward = 0
        info = {}
        truncated = False
        terminated = False

        return observation, reward, terminated, truncated, info

    def close(self):
        self.pyboy.stop(save=False)
        self.pyboy_state.close()
