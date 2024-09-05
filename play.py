import importlib

import envs.SuperJetPakEnv
from models.BaseModel import BaseModel
from models.DQN import DQN


def _get_model_class(name):
    module = importlib.import_module('models.' + name)
    return getattr(module, name)


def get_model(name, parameters) -> BaseModel:
    return _get_model_class(name)(parameters)


if __name__ == '__main__':
    env = envs.SuperJetPakEnv.SuperJetPakEnv('roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc',
                                             'roms/states/state1.state',
                                             force_gbc=False,
                                             ticks_per_action=4,
                                             force_discrete=True,
                                             flatten=False,
                                             grayscale=True)
    model = DQN.load('trained/convolution_dueling_gelu_multistates_per_v2')
    observation, info = env.reset()
    for x in range(10000):
        action = model.predict(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(action, reward, terminated, info)
    env.close()
