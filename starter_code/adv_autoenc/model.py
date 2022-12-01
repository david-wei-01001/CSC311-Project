import torch.nn as nn

import torch.utils.data

import torch


class Encoder(nn.Module):
    def __init__(self, k=100):
        super(Encoder, self).__init__()

        # Define linear functions.
        self.encoder = nn.Sequential(
            nn.Linear(1774, 256),
            nn.PReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, k),
        )

    def get_weight_norm(self):
        norm = 0
        for step in self.encoder:
            if type(step) == torch.nn.modules.linear.Linear:
                norm += torch.norm(step.weight, 2) ** 2
        return norm

    def forward(self, inputs):
        return self.encoder(inputs)


class Decoder(nn.Module):
    def __init__(self, k=100):
        super(Decoder, self).__init__()

        # Define linear functions.
        self.decoder = nn.Sequential(
            nn.Linear(k, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 1774),
        )

    def get_weight_norm(self):
        norm = 0
        for step in self.decoder:
            if type(step) == torch.nn.modules.linear.Linear:
                norm += torch.norm(step.weight, 2) ** 2
        return norm

    def forward(self, inputs):
        return self.decoder(inputs)


class Discriminator(nn.Module):
    def __init__(self, k=100):
        super(Discriminator, self).__init__()

        # Define linear functions.
        self.discriminator = nn.Sequential(
            nn.Linear(k, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 1),
        )

    def get_weight_norm(self):
        norm = 0
        for step in self.discriminator:
            if type(step) == torch.nn.modules.linear.Linear:
                norm += torch.norm(step.weight, 2) ** 2
        return norm

    def forward(self, inputs):
        return self.discriminator(inputs)