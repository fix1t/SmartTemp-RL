import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """
    A neural network for approximating action-value functions in reinforcement learning.

    The network architecture is simple, consisting of three fully connected layers.
    It takes an input representing the state of the environment and outputs a value for each action.

    Parameters:
        state_size (int): Dimension of each input state.
        action_size (int): Number of actions that can be taken in the environment.
        seed (int, optional): Random seed for initializing weights. Defaults to 42.
    """
    def __init__(self, state_size, action_size, seed=42):
        """
        Initializes the network with two hidden layers and one output layer.
        """
        super(Network, self).__init__()  # Initializes the base class nn.Module
        self.seed = torch.manual_seed(seed)  # Sets the random seed for PyTorch

        # Defines the first fully connected layer (input layer to hidden layer 1)
        self.fc1 = nn.Linear(state_size, 64)  # First hidden layer with 64 units

        # Defines the second fully connected layer (hidden layer 1 to hidden layer 2)
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer with 64 units

        # Defines the third fully connected layer (hidden layer 2 to output layer)
        self.fc3 = nn.Linear(64, action_size)  # Output layer with 'action_size' units

    def forward(self, state):
        """
        Defines the forward pass of the network.

        Parameters:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output tensor containing action values.
        """
        # Passes the input through the first layer and apply ReLU (rectified linear unit) activation function
        x = F.relu(self.fc1(state))

        # Passes the result through the second layer and apply ReLU (rectified linear unit) activation function
        x = F.relu(self.fc2(x))

        # Passes the result through the third layer (no activation function here, as this is the output layer)
        return self.fc3(x)
