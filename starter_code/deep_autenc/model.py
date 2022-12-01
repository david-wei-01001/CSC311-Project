import torch.nn as nn

import torch.utils.data

import torch


class AutoEncoder(nn.Module):
    def __init__(self, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

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
        self.decoder = nn.Sequential(
            nn.Linear(k, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 1774),
        )

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        norm = 0
        norm += torch.norm(self.encoder[0].weight, 2) ** 2
        norm += torch.norm(self.encoder[3].weight, 2) ** 2
        norm += torch.norm(self.encoder[6].weight, 2) ** 2
        norm += torch.norm(self.decoder[0].weight, 2) ** 2
        norm += torch.norm(self.decoder[2].weight, 2) ** 2
        norm += torch.norm(self.decoder[4].weight, 2) ** 2
        return norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return decoded
