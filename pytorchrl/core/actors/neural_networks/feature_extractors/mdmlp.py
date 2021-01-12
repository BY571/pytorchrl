import torch
import torch.nn as nn
from .utils import init

class MultiDiscreteMLP(nn.Module):
    """
    Multilayer Perceptron network for Multi Discrete Action Spaces

    Parameters
    ----------
    input_shape : tuple
        Shape input tensors.
    hidden_size : int
        Hidden layers sizes.


    Attributes
    ----------
    feature_extractor : nn.Module
        Neural network feature extractor block.
    """
    def __init__(self, input_shape, hidden_size=256, discrete_actions=9, out_size=3)
        super(MultiDiscreteMLP, self).__init__()

        self.layer1 = nn.Linear(input_shape, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.out_layer1 = nn.Linear(hidden_size, out_size)
        self.out_layer2 = nn.Linear(hidden_size, out_size)
        self.out_layer3 = nn.Linear(hidden_size, out_size)
        self.out_layer4 = nn.Linear(hidden_size, out_size)
        self.out_layer5 = nn.Linear(hidden_size, out_size)
        self.out_layer6 = nn.Linear(hidden_size, out_size)
        self.out_layer7 = nn.Linear(hidden_size, out_size)
        self.out_layer8 = nn.Linear(hidden_size, out_size)
        self.out_layer9 = nn.Linear(hidden_size, out_size)
        

    def forward(self, inputs):
        """ """
        x = torch.relu(self.layer1(inputs))
        x = torch.relu(self.layer2(x))

        # this is supposed to be the output layer... so in NN Base
        x1 = self.out_layer1(x)
        x2 = self.out_layer1(x)
        x3 = self.out_layer1(x)
        x4 = self.out_layer1(x)
        x5 = self.out_layer1(x)
        x6 = self.out_layer1(x)
        x7 = self.out_layer1(x)
        x8 = self.out_layer1(x)
        x9 = self.out_layer1(x)
