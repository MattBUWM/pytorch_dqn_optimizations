from torch import nn

from networks.Layers import NoisyLinearLayer


class DuelingNoisyFeedForwardNN(nn.Module):

    def __init__(self, in_dim, out_dim, activation=None):
        super(DuelingNoisyFeedForwardNN, self).__init__()
        if activation is None:
            self.activation1_adv = lambda x: x
            self.activation1_val = lambda x: x

            self.activation2_adv = lambda x: x
            self.activation2_val = lambda x: x
        else:
            activation_layer = getattr(nn, activation)
            self.activation1_adv = activation_layer()
            self.activation1_val = activation_layer()

            self.activation2_adv = activation_layer()
            self.activation2_val = activation_layer()

        self.layer1_adv = NoisyLinearLayer(in_dim[0], 256)
        self.layer1_val = NoisyLinearLayer(in_dim[0], 256)

        self.layer2_adv = NoisyLinearLayer(256, 128)
        self.layer2_val = nn.Linear(256, 128)

        self.layer3_adv = NoisyLinearLayer(128, out_dim)
        self.layer3_val = NoisyLinearLayer(128, 1)

    def forward(self, x):
        adv = self.layer1_adv(x)
        val = self.layer1_val(x)

        adv = self.activation1_adv(adv)
        val = self.activation1_val(val)

        adv = self.layer2_adv(adv)
        val = self.layer2_val(val)

        adv = self.activation2_adv(adv)
        val = self.activation2_val(val)

        adv = self.layer3_adv(adv)
        val = self.layer3_val(val)

        adv_mean = adv.mean(1, keepdim=True)

        x = val + adv - adv_mean

        return x
