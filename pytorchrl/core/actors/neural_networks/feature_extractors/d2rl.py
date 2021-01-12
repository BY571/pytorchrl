
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class D2RL(nn.Module):
    """
    D2RL: DEEP DENSE ARCHITECTURES IN REINFORCEMENT LEARNING
    https://arxiv.org/pdf/2010.09163.pdf

    Parameters
    ----------
    input_shape : tuple
        Shape input tensors.
    hidden_sizes : int
        Hidden layers sizes.

    Attributes
    ----------
    fc1 : nn.Module
        First dense fully connected neural network layer
    fc2 : nn.Module
        Second dense fully connected neural network layer
    fc3 : nn.Module
        Third dense fully connected neural network layer
    fc4 : nn.Module
        Fourth dense fully connected neural network layer

    """
    def __init__(self, input_shape, hidden_sizes=[256,256]):
        super(D2RL, self).__init__()

        hidden_size = 256
        in_dim = hidden_size+input_shape[0]
        self.fc1 = nn.Linear(input_shape[0], hidden_size)
        self.fc2 = nn.Linear(in_dim, hidden_size)
        self.fc3 = nn.Linear(in_dim, hidden_size)
        self.fc4 = nn.Linear(in_dim, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))

    def forward(self, inputs):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor
            Input data.

        Returns
        -------
        out : torch.tensor
            Output feature map.
        """
        x = F.relu(self.fc1(inputs))
        x = torch.cat([x, inputs], dim=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([x, inputs], dim=1)
        x = F.relu(self.fc3(x))
        x = torch.cat([x, inputs], dim=1)

        return F.relu(self.fc4(x))

