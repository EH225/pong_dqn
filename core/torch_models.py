"""
This module builds off the classes contained in core/base_components and defines a class instance for a linear
deep Q-network and a CNN-based deep Q-network that replicates the network archtecture of the Google DeepMind
model published in Nature.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import torch
import torch.nn as nn
from core.base_components import DQN
from utils.general import compute_img_out_dim


class LinearDQN(DQN):
    """
    Implementation of a single fully connected layer with Pytorch to be utilized in the DQN algorithm.
    """

    def initialize_models(self) -> None:
        """
        Initializes the 2 required models i.e. the Q network and the target Q network separately.

        The input to these networks will be an image of shape: (img_height * img_width, channels)
        with channels = n_channels * self.config["hyper_params"]["state_history"]

        n_channels are how many color channels are used for each image and we also include multiple recent
        frames stacked together to give the model a short recent history to condition on when making a
        Q-estimate and selecting an action.

        - self.q_network (torch model): Variable to store our q network implementation, this will be the
            actual value-function approximator that outputs Q^(s,a) values.
        - self.target_network (torch model): Variable to store our target network implementation, which will
            be the same as the q_network on a lagged update basis to be used for fixed Q and reply learning.
            Every so often, we will copy over the learned weights of self.q_network into self.target_network.
        """
        img_height, img_width, n_channels = self.env.observation_space.shape  # Unpack state size dimensions
        num_actions = self.env.action_space.n  # The number of possiable action we can select from
        # input_dim is the size of the flattened state vector that we will pass into the Q-network
        input_dim = img_height * img_width * n_channels * self.config["hyper_params"]["state_history"]

        self.q_network = nn.Linear(input_dim, num_actions)  # Create a linear layer that accepts in the
        # flattened state vector of size (input_dim) and outputs logits / scores for each possiable action
        self.optimizer = torch.optim.Adam(self.q_network.parameters())  # Set the optimizer for training this
        # model, note that we only train the q_network

        self.target_network = nn.Linear(input_dim, num_actions)  # Set self.target_network to be the same
        # configuration as self.q_netowrk, but initialized to be a different object in memory so that updates
        # to q_network are not also reflected here in target_network until we specifically copy them over.

    def get_q_values(self, state: torch.Tensor, network: str = "q_network") -> torch.Tensor:
        """
        Returns Q-values for all actions given the input state provided for each example in the batch.

        :param state: A torch.tensor of size (batch_size, state_history, img_height, img_width, nchannels)
            which encodes all the information of the current state i.e. what the game screend currently looks
            like and also what it looked like for a few frames prior.
        :param network: The name of the network, either "q_network" or "target_network" which specifies which
            network to use to obtain Q-values over all possiable actions.
        :return: Returns a tensor of size (batch_size, num_actions) with q-values for each action for each
            input state in the batch dimension.
        """
        # Flatten the input state into dimensions: (batch_size, img_h x img_w x n_channels x state_hist)
        # Pass the input img encoded as a state tensor through the linear layer to get the q-vals for all
        # actions for each example in the batch, the output of the linear layer is of size num_actions
        return getattr(self, network)(torch.flatten(state, start_dim=1))


class NatureDQN(DQN):
    """
    Implementation of DeepMind's Nature paper, see below for details:
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    The network archtecture is as follows:           The values below outline the dims used in the Pong env
        - (8 x 8) Conv2d -> Leaky ReLU activation    (N, C=1,  H=80, W=80) -> (N, C=32, H=20, W=20)
        - (4 x 4) Conv2d -> Leaky ReLU activation    (N, C=32, H=20, W=20) -> (N, C=64, H=9,  W=9)
        - (3 x 3) Conv2d -> Leaky ReLU activation    (N, C=64, H=9,  W=9)  -> (N, C=64, H=7,  W=7)
        - Flatten for passage into the FFNN
        - Linear Layer H=512 -> Leaky ReLU activation
        - Output Linear Layer with num_actions output nodes
    """

    def initialize_models(self):
        """
        Initializes the 2 required models i.e. the Q network and the target Q network separately.

        The input to these networks will be an image of shape: (img_height * img_width, channels)
        with channels = n_channels * self.config["hyper_params"]["state_history"]

        n_channels are how many color channels are used for each image and we also include multiple recent
        frames stacked together to give the model a short recent history to condition on when making a
        Q-estimate and selecting an action.

        - self.q_network (torch model): Variable to store our q network implementation, this will be the
            actual value-function approximator that outputs Q^(s,a) values.
        - self.target_network (torch model): Variable to store our target network implementation, which will
            be the same as the q_network on a lagged update basis to be used for fixed Q and reply learning.
            Every so often, we will copy over the learned weights of self.q_network into self.target_network.
        """
        img_height, img_width, n_channels = self.env.observation_space.shape  # Unpack state size dimensions
        num_actions = self.env.action_space.n  # The number of possiable action we can select from

        # Compute how the dimension of the images change as they pass through the convolutions so that we
        # have the required node count at the end for the final fully-connected layers, add padding to layer 1
        input_dim = (img_height, img_width)  # The initial input image dimensions going into the model
        out_dim_1 = compute_img_out_dim(input_dim, kernel_size=8, padding=2, dialation=1, stride=4)
        out_dim_2 = compute_img_out_dim(out_dim_1, kernel_size=4, padding=0, dialation=1, stride=2)
        out_dim_3 = compute_img_out_dim(out_dim_2, kernel_size=3, padding=0, dialation=1, stride=1)

        for model_name in ["q_network", "target_network"]:
            # Replicate the same structure for the q_network and target_network but saved as different objects
            model = nn.Sequential(
                nn.Conv2d(in_channels=n_channels * self.config["hyper_params"]["state_history"],
                          out_channels=32, kernel_size=(8, 8), stride=4, padding=2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
                nn.LeakyReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(out_dim_3[0] * out_dim_3[1] * 64, 512),
                nn.LeakyReLU(),
                nn.Linear(512, num_actions)
            )
            setattr(self, model_name, model)

        self.optimizer = torch.optim.Adam(self.q_network.parameters())  # Set the optimizer for training this
        # model, note that we only train the q_network

    def get_q_values(self, state: torch.Tensor, network: str) -> torch.Tensor:
        """
        Returns Q-values for all actions given the input state provided for each example in the batch.

        :param state: A torch.tensor of size (batch_size, state_history, img_height, img_width, nchannels)
            which encodes all the information of the current state i.e. what the game screend currently looks
            like and also what it looked like for a few frames prior.
        :param network: The name of the network, either "q_network" or "target_network" which specifies which
            network to use to obtain Q-values over all possiable actions.
        :return: Returns a tensor of size (batch_size, num_actions) with q-values for each action for each
            input state in the batch dimension.
        """
        batch_size, k, img_h, img_w, img_c = state.shape  # Unpack to get dimensions
        # Merge the n_channels and frame_history into stacked 1 dimension
        state = torch.permute(state, (0, 2, 3, 4, 1)).reshape(batch_size, img_h, img_w, -1)
        # The input to the Conv2d layers must be (batch_size, in_channels, img_height, img_width) so we
        # permute the dimensons 1 more time to move the n_channels x frame_history to index 1
        return getattr(self, network)(torch.permute(state, (0, 3, 1, 2)))  # (batch_size, num_actions)
