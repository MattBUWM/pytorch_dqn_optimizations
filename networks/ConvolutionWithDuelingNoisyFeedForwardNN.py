from torch import nn
import torch.nn.functional as F

from networks.Layers import NoisyLinearLayer

class ConvolutionWithDuelingFeedForwardNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ConvolutionWithDuelingFeedForwardNN, self).__init__()
        conv_kernel = (8, 8)
        pool_kernel = (2, 2)
        channels_mult = 5
        self.out_dim = out_dim
        self.image_size = [in_dim[1], in_dim[2]]
        self.conv1 = nn.Conv2d(in_channels=in_dim[0], out_channels=in_dim[0] * channels_mult, kernel_size=conv_kernel,
                               stride=1)
        self.image_size = [self.image_size[0] - conv_kernel[0] + 1, self.image_size[1] - conv_kernel[1] + 1]
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel, stride=2)
        self.image_size = [int(self.image_size[0] / pool_kernel[0]), int(self.image_size[1] / pool_kernel[1])]

        conv_kernel = (4, 4)
        pool_kernel = (2, 2)

        self.conv2 = nn.Conv2d(in_channels=in_dim[0] * channels_mult, out_channels=in_dim[0] * channels_mult,
                               kernel_size=conv_kernel, stride=1)
        self.image_size = [self.image_size[0] - conv_kernel[0] + 1, self.image_size[1] - conv_kernel[1] + 1]
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel, stride=2)
        self.image_size = [int(self.image_size[0] / pool_kernel[0]), int(self.image_size[1] / pool_kernel[1])]

        self.final_size = self.image_size[0] * self.image_size[1] * in_dim[0] * channels_mult

        self.layer1_adv = NoisyLinearLayer(self.final_size, 256)
        self.layer1_val = NoisyLinearLayer(self.final_size, 256)

        self.layer2_adv = NoisyLinearLayer(256, 128)
        self.layer2_val = NoisyLinearLayer(256, 128)

        self.layer3_adv = NoisyLinearLayer(128, out_dim)
        self.layer3_val = NoisyLinearLayer(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        adv = F.relu(self.layer1_adv(x.reshape(-1, self.final_size)))
        val = F.relu(self.layer1_val(x.reshape(-1, self.final_size)))

        adv = F.relu(self.layer2_adv(adv))
        val = F.relu(self.layer2_val(val))

        adv = F.relu(self.layer3_adv(adv))
        val = F.relu(self.layer3_val(val))

        adv_mean = adv.mean(1)
        adv_mean = adv_mean.unsqueeze(1)
        adv_mean = adv_mean.expand(x.size(0), self.out_dim)

        # print(adv, adv_mean)

        x = val + adv - adv_mean

        return x
