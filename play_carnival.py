import importlib

import gymnasium as gym

from models.BaseModel import BaseModel
from models.DQN import DQN


def _get_model_class(name):
    module = importlib.import_module('models.' + name)
    return getattr(module, name)


def get_model(name, parameters) -> BaseModel:
    return _get_model_class(name)(parameters)


if __name__ == '__main__':
    flatten = False
    env = gym.make('ALE/Boxing-v5', obs_type='grayscale', render_mode='human')
    env = gym.wrappers.FrameStack(env, 4)
    if flatten:
        env = gym.wrappers.FlattenObservation(env)

    model = DQN.load('trained/boxing_convolution_dueling_gelu')
    observation, info = env.reset()
    env.render()
    for x in range(10000):
        action = model.predict(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(action, reward, terminated, info)
    env.close()
