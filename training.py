import importlib

from models.BaseModel import BaseModel
import envs.SuperJetPakEnv


def _get_model_class(name):
    module = importlib.import_module('models.' + name)
    return getattr(module, name)


def get_model(name, parameters) -> BaseModel:
    return _get_model_class(name)(parameters)


if __name__ == '__main__':
    env = envs.SuperJetPakEnv.SuperJetPakEnv('roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc',
                                             'roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc.state',
                                             force_gbc=False,
                                             ticks_per_action=4,
                                             force_discrete=True,
                                             flatten=False,
                                             grayscale=True)

    model_parameters = {
        'model_path': 'test',
        'optimizer_parameters': {
            'optimizer': 'AdamW',
            'lr': 1e-4,
            'amsgrad': True
        },
        'network': 'ConvolutionWithDuelingFeedForwardNN',
        'obs_shape': env.observation_space.shape,
        'action_shape': int(env.action_space.n),
        'target_epoch': 50,
        'batch_size': 128,
        'save_freq': 5,
        'replay_memory_capacity': 2500,
        'gamma': 0.98,
        'tau': 0.0075,
        'epsilon': {
            'start': 0.9,
            'end': 0.05,
            'decay': 25000,
        }
    }
    model = get_model('DQN', model_parameters)
    model.train(env)
    model.save()
    env.close()
