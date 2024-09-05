import importlib

from models.BaseModel import BaseModel
import envs.SuperJetPakEnv


def _get_model_class(name):
    module = importlib.import_module('models.' + name)
    return getattr(module, name)


def get_model(name, parameters, per=False) -> BaseModel:
    return _get_model_class(name)(parameters, per=per)


if __name__ == '__main__':
    env = envs.SuperJetPakEnv.SuperJetPakEnv('roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc',
                                             'roms/states',
                                             force_gbc=False,
                                             ticks_per_action=4,
                                             force_discrete=True,
                                             flatten=False,
                                             grayscale=True)

    model_parameters = {
        'model_path': 'trained/convolution_dueling_gelu_multistates_per_v2',
        'optimizer_parameters': {
            'optimizer': 'AdamW',
            'lr': 1e-5,
            'amsgrad': True
        },
        'network': 'ConvolutionWithDuelingFeedForwardNN',
        'activation_function': 'GELU',
        'obs_shape': env.observation_space.shape,
        'action_shape': int(env.action_space.n),
        'target_episode': 100,
        'batch_size': 256,
        'save_freq': 5,
        'replay_memory_capacity': 10000,
        'gamma': 0.99,
        'tau': 0.002,
        'epsilon': {
            'start': 0.9,
            'end': 0.05,
            'decay': 10000,
        }
    }
    model = get_model('DQN', model_parameters, per=True)
    model.train(env)
    model.save()
    env.close()
