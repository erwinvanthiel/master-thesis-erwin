"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn

class AuxiliaryClassifier(nn.Module):
    def __init__(self, channels_img, output_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.conv1 = self._block(channels_img, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = self._block(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64*16*16, 1024, bias=True)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, output_classes, bias=True)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, 64*16*16)
        x = self.fc1(x)
        x = self.dropout(x)
        aux = self.fc2(x)
        return aux


class TargetClassifier(nn.Module):
    def __init__(self, channels_img, output_classes):
        super(TargetClassifier, self).__init__()
        self.conv1 = self._block(channels_img, 64, 5, 1, 1)
        self.conv2 = self._block(64, 64, 5, 1, 1)
        self.fc1 = nn.Sequential(nn.Linear(64*60*60, 128, bias=True), nn.ReLU())
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, output_classes, bias=True)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = x.view(-1, 64*60*60)
        x = self.fc1(x)
        x = self.dropout2(x)
        classes = self.fc2(x)
        return classes



class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# def test():
#     N, in_channels, H, W = 8, 3, 64, 64
#     noise_dim = 100
#     x = torch.randn((N, in_channels, H, W))
#     disc = Discriminator(in_channels, 8)
#     assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
#     gen = Generator(noise_dim, in_channels, 8)
#     z = torch.randn((N, noise_dim, 1, 1))
#     assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# test()
