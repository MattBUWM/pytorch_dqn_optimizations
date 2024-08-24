import importlib

from models.BaseModel import BaseModel
import gymnasium as gym


def _get_model_class(name):
    module = importlib.import_module('models.' + name)
    return getattr(module, name)


def get_model(name, parameters) -> BaseModel:
    return _get_model_class(name)(parameters)


if __name__ == '__main__':
    flatten = False
    env = gym.make('ALE/Boxing-v5', obs_type='grayscale')
    env = gym.wrappers.FrameStack(env, 4)
    if flatten:
        env = gym.wrappers.FlattenObservation(env)

    print(env.observation_space.shape)
    model_parameters = {
        'model_path': 'trained/boxing_convolution_dueling_gelu',
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
        'save_freq': 10,
        'replay_memory_capacity': 10000,
        'gamma': 0.97,
        'tau': 0.005,
        'epsilon': {
            'start': 0.9,
            'end': 0.05,
            'decay': 10000,
        }
    }
    model = get_model('DQN', model_parameters)
    model.train(env)
    model.save()
    env.close()
