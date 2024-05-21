import importlib
import os

from torch import nn, device, cuda

SupportedOptims = ['AdamW', 'Adam']


def _get_network_class(name):
    module = importlib.import_module('networks.' + name)
    return getattr(module, name)


def get_network(name, in_dim, out_dim, activation) -> nn.Module:
    return _get_network_class(name)(in_dim, out_dim, activation=activation)


class BaseModel:
    def __init__(self, parameters, load_existing=False):
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model_path = parameters['model_path']
        if os.path.exists(self.model_path) and not load_existing:
            raise FileExistsError('model with this path already exists')
        os.makedirs(self.model_path, exist_ok=True)
        self.optimizer_parameters = parameters['optimizer_parameters']
        if self.optimizer_parameters['optimizer'] not in SupportedOptims:
            raise ValueError('optimizer not recognized or supported')
        self.network_type = parameters['network']
        self.activation_function = parameters['activation_function']
        self.target_epoch = parameters['target_epoch']
        if load_existing:
            self.current_epoch = parameters['current_epoch']
            self.steps_done = parameters['steps_done']
        else:
            self.current_epoch = 0
            self.steps_done = 0
        self.save_freq = parameters['save_freq']

    def train(self, env):
        for epoch in range(self.target_epoch - self.current_epoch):
            self.training_epoch(env)
            self.current_epoch += 1
            if self.current_epoch % self.save_freq == 0:
                self.save()

    def training_epoch(self, env):
        raise NotImplementedError

    def predict(self, observation, env, training=True):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    @staticmethod
    def load(model_path):
        raise NotImplementedError



