"""
    File: network.py
    Author: Gabriel Biel

    Description: A customizable Feed Forward Neural Network.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    """
    A customizable Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim, hidden_layers=[64, 64], activation=nn.ReLU, output_activation=nn.Softmax(dim=-1), seed=42):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim (int): input dimensions
            out_dim (int): output dimensions
            seed (int, optional): Random seed for initializing weights. Defaults to 42.
            hidden_layers (list): list of integers, where each integer is the number of neurons in a layer
            activation (callable): activation function to use between hidden layers
            output_activation (callable): activation function for the output layer

        Return:
            None
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)  # Sets the random seed for PyTorch

        # Creating the network layers dynamically based on hidden_layers
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        # Adding the output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_activation is not None:
            layers.append(output_activation())

        # Combine all layers into a Sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input

        Return:
            output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        return self.layers(obs)
