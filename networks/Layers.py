import math

import torch
from torch import nn


class NoisyLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, init_sigma=0.017):
        super(NoisyLinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_mu = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.bias_mu = nn.Parameter(torch.empty(self.output_size))

        self.weight_sigma = nn.Parameter(torch.full((self.output_size, self.input_size), init_sigma))
        self.bias_sigma = nn.Parameter(torch.full((self.output_size,), init_sigma))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    def forward(self, x):
        epsilon_i = torch.randn(self.input_size, device=x.device)
        epsilon_j = torch.randn(self.output_size, device=x.device)

        weight_epsilon = torch.sign(epsilon_j).mul(torch.sqrt(torch.abs(epsilon_j))).outer(
            torch.sign(epsilon_i).mul(torch.sqrt(torch.abs(epsilon_i))))
        bias_epsilon = torch.sign(epsilon_j).mul(torch.sqrt(torch.abs(epsilon_j)))

        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        bias = self.bias_mu + self.bias_sigma * bias_epsilon

        return nn.functional.linear(x, weight, bias)
