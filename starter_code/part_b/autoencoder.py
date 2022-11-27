import torch.nn as nn

import torch.utils.data

import torch


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.encoder = nn.Sequential(
            nn.Linear(num_question, 887),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(887, 221),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(221, k),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 221),
            nn.PReLU(),
            nn.Linear(221, 887),
            nn.PReLU(),
            nn.Linear(887, num_question),
            nn.Sigmoid(),
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
        #####################################################################
        #                                                                   #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decoded
