from torch import nn


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(FeedForwardNN, self).__init__()
        if activation is None:
            self.activation1 = lambda x: x
            self.activation2 = lambda x: x
        else:
            activation_layer = getattr(nn, activation)
            self.activation1 = activation_layer()
            self.activation2 = activation_layer()
        print(in_dim)
        self.layer1 = nn.Linear(in_dim[0], 256)

        self.layer2 = nn.Linear(256, 128)

        self.layer3 = nn.Linear(128, out_dim)

    def forward(self, x):

        x = self.layer1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)

        x = self.layer3(x)

        return x
