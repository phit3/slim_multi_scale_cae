import logging
import typing
import torch
import torch.nn as nn

# import torch summary
from torchsummary import summary


class View(torch.nn.Module):
    ''' Reshapes input tensor.'''
    def __init__(self, target_shape):
        super().__init__()
        self.shape = target_shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, inputs):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = inputs.size(0)
        shape = (batch_size, *self.shape)
        out = inputs.view(shape)
        return out

class BaseCAE(torch.nn.Module):
    class Encoder(torch.nn.Module):
        @property
        def channels(self):
            return [self.channel_factor, self.channel_factor * 2, self.channel_factor * 4, self.channel_factor * 8]

        def __init__(self, width: int, height: int, channel_factor: int, latent_dim: int, n_stages: int = 1):
            super().__init__()
            self.width = width
            self.height = height
            self.channel_factor = channel_factor
            self.latent_dim = latent_dim
            self.n_stages = 1
            self.latent_shape = [0, 0, 0]
            pool_size = (2, 2)
            kernel_size = (3, 3)
            self.do_p = 0.1
            input_channels = 1
            activation = nn.LeakyReLU

            pre = self.channels[-1] * int(self.width / (2 ** len(self.channels))) * int(self.height / (2 ** len(self.channels)))
            pre_red_1 = int((pre + self.latent_dim) / 2)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=self.channels[0], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[0]),
                nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[0], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[0]),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
                nn.Dropout(self.do_p),

                nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[1]),
                nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[1]),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
                nn.Dropout(self.do_p),

                nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[2]),
                nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[2]),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
                nn.Dropout(self.do_p),

                nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[3]),
                nn.Conv2d(in_channels=self.channels[3], out_channels=self.channels[3], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[3]),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
                nn.Dropout(self.do_p),

                nn.Flatten(),

                nn.Linear(pre, pre_red_1),
                activation(),
                nn.BatchNorm1d(pre_red_1),

                nn.Linear(pre_red_1, self.latent_dim),
                nn.Tanh()
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.encoder(inputs)
            return out

    class Decoder(torch.nn.Module):
        class UpOperation(nn.Module):
            def __init__(self, channel, scale_factor):
                super().__init__()
                self.scale_factor = scale_factor
                self.up = nn.Upsample(scale_factor=self.scale_factor)

            def forward(self, x):
                return self.up(x)

        @property
        def channels(self):
            return [self.channel_factor * 8, self.channel_factor * 4, self.channel_factor * 2, self.channel_factor]

        def __init__(self, width: int, height: int, channel_factor: int, latent_dim: int):
            super().__init__()
            self.width = width
            self.height = height
            self.channel_factor = channel_factor
            self.latent_dim = latent_dim
            activation = nn.LeakyReLU
            self.input_channels = 1
            us_factor = 2
            kernel_size = (3, 3)
            self.do_p = 0.1

            pre = self.channels[0] * int(self.width / (2 ** len(self.channels))) * int(self.height / (2 ** len(self.channels)))
            pre_red_1 = int((pre + self.latent_dim) / 2)
            
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, pre_red_1),
                activation(),
                nn.BatchNorm1d(pre_red_1),

                nn.Linear(pre_red_1, pre),
                activation(),
                nn.BatchNorm1d(pre),

                View(target_shape=(self.channels[0], int(self.height / (2**len(self.channels))), int(self.width / (2**len(self.channels))))),

                self.UpOperation(channel=None, scale_factor=us_factor),
                nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[1]),
                nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[1]),
                nn.Dropout(self.do_p),

                self.UpOperation(channel=None, scale_factor=us_factor),
                nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[2]),
                nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[2]),
                nn.Dropout(self.do_p),

                self.UpOperation(channel=None, scale_factor=us_factor),
                nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[3]),
                nn.Conv2d(in_channels=self.channels[3], out_channels=self.channels[3], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[3]),
                nn.Dropout(self.do_p),

                self.UpOperation(channel=None, scale_factor=us_factor),
                nn.Conv2d(in_channels=self.channels[3], out_channels=self.channels[-1], kernel_size=kernel_size, stride=(1, 1), padding='same'),
                activation(),
                nn.BatchNorm2d(self.channels[-1]),
                nn.Conv2d(in_channels=self.channels[-1], out_channels=self.input_channels, kernel_size=kernel_size, stride=(1, 1), padding='same'),
                nn.Sigmoid()
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.decoder(inputs)

    def __init__(self, width: int, height: int, channel_factor: int, latent_dim: int):
        super().__init__()
        self.encoder = BaseCAE.Encoder(width=width, height=height, channel_factor=channel_factor, latent_dim=latent_dim)
        self.decoder = BaseCAE.Decoder(width=width, height=height, channel_factor=channel_factor, latent_dim=latent_dim)
        # print the model summary
#        summary(self.encoder.cuda(), (1, width, height))
#        summary(self.decoder.cuda(), (latent_dim, ))
    def next_stage(self):
        return False

    def forward(self, inputs: torch.Tensor, mode='train') -> torch.Tensor:
        self.lat_rep = self.encoder(inputs)
        return self.decoder(self.lat_rep), self.lat_rep

