"""
This module contains the base components of training a deep Q-network model. It defines linear schedule
objects for decaying the exploration parameter (epsilon) over time and a general set of classes for training
Q-networks and deep Q-networks.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import time
import numpy as np
from typing import Callable, List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import ale_py
from collections import deque, defaultdict
from utils.general import get_logger, Progbar
from utils.replay_buffer import ReplayBuffer

from utils.general import read_yaml, pong_img_transform, save_eval_scores

gym.register_envs(ale_py)


#########################################
### Exploration Rate Decay Schedulers ###
#########################################
# TODO: Section marker

class LinearSchedule:
    """
    Sets a linear schedule for the decay of the exploration parameter (epsilon) over time.
    """

    def __init__(self, param_begin: float, param_end: float, nsteps: int):
        """
        Initializes a LinearSchedule object instance with an update() method that will update a parameter
        (e.g. a learning rate or epsilon exploration rate) being tracked at self.param.

        :param eps_begin: The exploration parameter epsilon's starting value.
        :param eps_end: The exploration parameter epsilon's ending value.
        :param nsteps: The number of steps over which the exploration parameter epsilon will decay from
            eps_begin to eps_end linearly.
        :returns: None
        """
        # msg = f"Param begin ({param_begin}) needs to be greater than or equal to end ({param_end})"
        # assert param_begin >= param_end, msg
        self.param = param_begin  # epsilon beings at eps_begin
        self.param_begin = param_begin
        self.param_end = param_end
        self.nsteps = nsteps
        # Using a linear decay schedule, the amount of decay for each timestep will be equal each time so
        # we can pre-compute the size of each decay step and store that here
        self.update_per_step = ((self.param_end - self.param_begin) / self.nsteps)

    def update(self, t: int) -> None:
        """
        Updates param internally at self.param using a linear interpolation from self.param _begin to
        self.param_end as t goes from 0 to self.nsteps. For t > self.nsteps self.param remains constant as
        the last updated self.param value, which is self.param_end.

        :param t: The time index i.e. frame number of the current step.
        :return: The updated exploration parameter value.
        """
        if t < self.nsteps:  # Prior to the end of the decay schedule, compute the linear decay +1 step
            self.param = self.param_begin + self.update_per_step * t
        else:  # After nsteps, set param to param_end
            self.param = self.param_end
        return self.param


class LinearExploration(LinearSchedule):
    """
    Implements an e-greedy exploration strategy with a linear exploration parameter (epsilon) decay.
    """

    def __init__(self, env, eps_begin: float, eps_end: float, nsteps: int):
        """

        :param env: A gym environment i.e. contains information about the action and state space.
        :param eps_begin: The exploration parameter epsilon's starting value.
        :param eps_end: The exploration parameter epsilon's ending value.
        :param nsteps: The number of steps over which the exploration parameter epsilon will decay from
            eps_begin to eps_end linearly.
        :return: None
        """
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)
        self.env = env

    def get_action(self, best_action: int) -> int:
        """
        Returns a randomly selected action with probability self.epsilon, otherwise returns the best_action
        according to the input kwarg provided.

        :param: best_action: An integer denoting the best action according to some policy.
        :return: An integer denoting an action.
        """
        if np.random.rand() < self.param:  # With probability (epsilon) return a randomly selected action
            return self.env.action_space.sample()
        else:  # With probability (1 - epsilon), return the best_action
            return best_action


#################################
### Deep Q-Network Definition ###
#################################
# TODO: Section marker

class DQN:
    """
    Base-class for implementing a Deep Q-Network RL model.
    """

    def __init__(self, env, config: dict, logger=None):
        """
        Initialize a Q Network and env.

        :param env: A gym environment i.e. contains information about the action and state space.
        :param config: A config dictionary read from yaml that specifies hyperparameters.
        :param logger: A logger instance from the logging module for screen updates during training.
        :return: None.
        """
        # Configure the directory for training outputs
        os.makedirs(config["output"]["output_path"], exist_ok=True)

        # Store the hyperparams and other inputs
        self.env = env
        self.config = config
        self.logger = logger if logger is not None else get_logger(config["output"]["log_path"])

        # These are to be defined when self.initialize_models() is called
        self.q_network, self.target_network, self.optimizer = None, None, None

        # Auto-detect which device should be used by the model by what hardware is available
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:  # Default to using the CPU if no GPU accelerator
            self.device = "cpu"

        # Call the build method to instantiate the needed variables for the RL model
        self.build()

        # Configure a summary writer from TensorBoard for tracking the progress of training as we go
        self.summary_writer = SummaryWriter(self.config["output"]["tensorboard"], max_queue=1e5)

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
        # This method is to be defined by an object in the torch_models module
        raise NotImplementedError

    def build(self):
        """
        Builds the models and performs necessary pre-processing steps
        1. Calls self.initialize_models() to instantiate the q_network and target_network models
        2. Loads in pre-trained weights if any are detected or randomly initializes them
        3. Moves the torch models to the appropriate device
        4. Compiles the model if specified in the config file
        """
        # 1). Initialize the q_network and target_network models
        self.initialize_models()

        # 2). Load in existing pre-trained weights if they are available and specified by the config
        if "load_dir" in self.config["model_training"].keys():
            load_dir = self.config["model_training"]["load_dir"]
            self.logger.info(f"Looking for existing model weights and optimizer in: {load_dir}")
            if os.path.exists(load_dir):  # Check that the load weights / optimizer directory is valid

                # A). Attempt to load in pre-trained model weights from disk if they are available
                wts_path = os.path.join(load_dir, "model.bin")
                if os.path.exists(wts_path):  # Check if there is a cached model weights file
                    # We only load weights for the q_network since the ones of target_network are copied in
                    # from the q_network periodically
                    wts = torch.load(wts_path, map_location=lambda storage, loc: storage, weights_only=True)
                    self.q_network.load_state_dict(wts)  # Load in the model weights to the q_network
                    self.logger.info("Existing model weights loaded successfully!")

                # B). Attempt to load in an existing optimizer state if one is available
                opt_path = os.path.join(load_dir, "model.optim.bin")
                if os.path.exists(opt_path):  # Check if there is a cached model optimizer file
                    self.optimizer.load_state_dict(torch.load(opt_path, weights_only=True))
                    self.logger.info("Existing optimizer weights loaded successfully!")
            else:
                self.logger.info("load_dir is not a valid directory")

        # NOTE: The code below is not necessary, the weights and biases are auto-initialized by PyTorch
        # else: # Otherwise if we're not using pre-trained weights, then initialize them randomly
        #     print("Initializing parameters randomly")

        #     def init_weights_randomly(m):
        #         if hasattr(m, "weight"):
        #             nn.init.xavier_uniform_(m.weight, gain=2 ** (1.0 / 2))
        #         if hasattr(m, "bias"):
        #             nn.init.zeros_(m.bias)

        #     self.q_network.apply(init_weights_randomly)

        self.update_target_network()  # Copy over the parameters from self.q_network to target_network

        # 3). Now that we have the models and their weights initialized, move them to the appropriate device
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)

        # 4). Check if the model should be compiled or not, if so then attempt to do so
        if self.config["model_training"].get("compile", False):
            try:
                compile_mode = self.config["model_training"]["compile_mode"]
                self.q_network = torch.compile(self.q_network, mode=compile_mode)
                self.target_network = torch.compile(self.target_network, mode=compile_mode)
                print("Models compiled")
            except Exception as err:
                print(f"Model compile attempted, but not supported: {err}")

    def get_q_values(self, state: torch.Tensor, network: str = "q_network") -> torch.Tensor:
        """
        Returns Q values for all actions given the input state provided.

        :param state: A torch.Tensor of size (batch_size, img_height, img_width, nchannels * state_history)
            which encodes all the information of the current state i.e. what the screen looks like currently
            and for a few prior frames.
        :param network: The name of the network, either "q_network" or "target_network" which specifies which
            network to use to obtain Q-values over all possible actions.
        :return: Returns a tensor of size (batch_size, num_actions) with q-values for each action for each
            input state in the batch dimension.
        """
        # This method is to be defined by an object in the torch_models module
        raise NotImplementedError

    def update_target_network(self) -> None:
        """
        This update_target method is called periodically to copy self.q_network weights to
        self.target_network. The target_network is a lagged version of the q_network. By freezing the
        parameters of the target_network during training for a sequence of steps, we reduce the variance
        of the model during training.
        """
        # Synchronize the weights of both networks, perform a "hard" copy of the weights from q to target
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_best_action(self, state: torch.tensor, default: int = None) -> Tuple[int, torch.tensor]:
        """
        This method is called by the train() method to generate the best action at a given timestep according
        to the current parameters of the q_network model and return the Q-values for each action estimated by
        the model.

        :param state: A frame-stacked state observations where the pixel values are encoded as float
            [0, 1] as a torch.tensor of size i.e. (batch_size=1, frame_stack, height, width, n_channels).
        :param default: A default action to take if state is None. Will return a randomly selected action
            if both state and default are None.
        :returns: A tuple of 2 elements:
            - The best action according to the model as an int
            - The estimated Q-values for all possible actions (with the selected being the argmax)
        """
        if state is None:
            action = default if default is not None else self.env.action_space.sample()
            return action, torch.zeros(self.env.action_space.n)

        with torch.no_grad():  # Gradient tracking not needed for this step, used to generate data
            # Use our learned q_network to estimate the Q-value of all possible actions
            q_values = self.get_q_values(state, "q_network")
        action = q_values.argmax().item()  # Select the argmax as the best action according to the model
        return action, q_values

    # def get_action(self, state: torch.tensor) -> int:
    #     """
    #     Returns an action with a soft-epsilon selection strategy so that every action has a non-zero
    #     probability of being selected:
    #         - With probability soft_epsilon, we select an action at random.
    #         - With probability (1 - soft_epsilon), we select the best action according to the model.

    #     :param state: A frame-stacked state observations i.e. (frame_stack, height, width, n_channels) where
    #         the pixel values are encoded as floats [0, 1] as a torch.tensor.
    #     :return: An action encoded as an int.
    #     """
    #     if state is None: # If None, then randomly sample an action
    #         return self.env.action_space.sample()

    #     # With probability epsilon, select an action at random
    #     if np.random.random() < self.config["model_training"]["soft_epsilon"]:
    #         return self.env.action_space.sample()
    #     else: # With probability (1 - epsilon) return the best action according to the model
    #         return self.get_best_action(state)[0] # Return just the action, drop the Q-values

    @property
    def policy(self) -> Callable:
        """
        Returns a function that maps states to actions i.e. model.policy(state) = action
        """
        return lambda state: self.get_best_action(state)[0]

    def save(self):
        """
        Saves the parameters of self.q_network to the model_output directory specified in the config file
        along with the current state of the optimizer if any.
        """
        save_dir = self.config["output"]["model_output"]
        os.makedirs(save_dir, exist_ok=True)  # Make dir if needed
        if self.q_network is not None:  # If a q_network has been instantiated, save its weights
            torch.save(self.q_network.state_dict(), os.path.join(save_dir, "model.bin"))
        if self.optimizer is not None:  # If an optimizer has been instantiated, save its weights
            torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "model.optim.bin"))

    def init_averages(self):
        """
        Defines extra attributes for monitoring the training with tensorboard.
        """
        self.avg_reward, self.max_reward, self.eval_reward = -21.0, -21.0, -21.0
        self.avg_q, self.max_q, self.std_q, self.std_reward = 0, 0, 0, 0

    def update_averages(self, rewards: deque, max_q_values: deque, q_values: deque,
                        scores_eval: list) -> None:
        """
        Updates the rewards averages and other summary stats for tensorboard.

        :param rewards: A deque of recent reward values.
        :param max_q_values: A deque of max q-values.
        :param q_values: A deque of recent q_values.
        :param scores_eval: A list of recent evaluation scores.
        :return: None.
        """
        if len(rewards) > 0:
            self.avg_reward = np.mean(rewards)  # Record the mean of recent rewards
            self.max_reward = np.max(rewards)  # Record the max of recent rewards
            self.std_reward = np.std(rewards)  # Record the std of recent rewards

        self.avg_q = np.mean(q_values)  # Record the mean of recent q-values
        self.max_q = np.mean(max_q_values)  # Record the mean of recent max q-values
        self.std_q = np.std(q_values)  # Record the std of recent rewards

        if len(scores_eval) > 0:  # If we have computed at least 1 evaluation score
            self.eval_reward = scores_eval[-1][1]  # Record the most recent evaluation score

    def add_summary(self, latest_loss, latest_total_norm, t):
        """
        Configurations for Tensorboard.
        """
        self.summary_writer.add_scalar("loss", latest_loss, t)
        self.summary_writer.add_scalar("grad_norm", latest_total_norm, t)
        self.summary_writer.add_scalar("Avg_Reward", self.avg_reward, t)
        self.summary_writer.add_scalar("Max_Reward", self.max_reward, t)
        self.summary_writer.add_scalar("Std_Reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg_Q", self.avg_q, t)
        self.summary_writer.add_scalar("Max_Q", self.max_q, t)
        self.summary_writer.add_scalar("Std_Q", self.std_q, t)
        self.summary_writer.add_scalar("Eval_Reward", self.eval_reward, t)

    def calc_loss(self, q_values_1: torch.Tensor, q_values_2: torch.tensor,
                  target_q_values_2: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                  terminated_mask: torch.Tensor, truncated_mask: torch.Tensor, wts: torch.Tensor
                  ) -> torch.float:
        """
        Calculates the MSE loss of a batch of inputs. The loss for an example is defined as:
            loss = (Q_samp(s) - Q(s, a))^2 = (y - y_hat)^2

        where Q_samp(s) using the standard DQN loss is computed as:
            Q_samp(s) = r + gamma * MAX_{a'}[Q_target(s', a')] or r if terminated

        and Q_samp(s) using the double DQN loss is comptued as:
            Q_samp(s) = r + gamma * [Q_target(s', a')] or r if terminated
            with a* = argmax_{a'}[Q_network(s', a')]

        When truncated is True, then then we can still run further steps in the env after the next state so
        we will still estimate the return thereafter using the target_network Q-values. For a terminated flag,
        that means no further states follow the next state so the only rewards obtained by the agent is the
        immediate reward going from s -> s' since nothing follows after s'. We therefore use bootstrapping
        function approximation in the case of truncated but not for termianted.

        For the standard DQN loss, the next action to take in the next state s' is determined by the
        target_network while in the double DQN loss, the q_network determines it while the target_network
        estimates the value of it. This decouples selection and evaluation, reducing over estimation of the
        Q-values during training and is generally better.

        In this loss we use the q_network to generate a series of q_values_1 for each action from the current
        state (s), and then select an action to take in the env based on those q_values (and some exploration
        allowance), which is recorded in actions. The env gives the agent a reward (recorded in rewards) and
        then the agent enters a new state s'. The value of each possible action (a') in the the next state
        (s') is computed using the target_network (target_q_values_2) or q_network (q_values_2) and is used
        to select the best action from the next state s'. In either case, the value of Q(s', a') is estimated
        using the target_q_values_2 values, which act as an estimate of what value one would expect optimally
        from the env thereafter starting in s'.

        The realized return (r) from taking action (a) from the current state (s) plus the expected discounted
        future return from starting thereafter in the next state (s') estimated by target_network is used to
        compute Q_samp(s) which acts as our y-value and is derived from our sample data. These are what we
        hope to learn to predict with our q_network model. These values are computed using a 1 step Bellman
        backup and therefore tend to be more accurate, but are also based on estimates from the
        target_network.

        We compare Q_samp with the values of Q(s, a) from the q_network computed at the starting state s.
        These are the "y_hat" values i.e. what our model predicts is the value from taking an action (a)
        starting from a the current state (s). In essence, if q_network is good, then it will be able to
        accurately predict Q(s, a) and that will match with what we actually experience after 1 timestep.

        With prioritized experience replay (PER), we over-sample data points from the replay buffer with the
        highest TD errors historically, but this will lead to biased gradients, unless we also use importance
        sampling to re-weight them when computing our loss which is the purpose of using the wts vector to
        compute the weighted MSE.

        :params q_values_1: torch.tensor with shape = (batch_size, num_actions)
            The Q-values that the current q_network estimates for taking action (a) from the current state (s)
            for each example in the batch (i.e. Q(s, a) for all a).
        :param q_values_2: torch.tensor with shape = (batch_size, num_actions) or None
            The Q-values that the current q_network estimates for taking action (a') from the next state (s')
            for each example in the batch (i.e. Q(s', a') for all a').
        :param target_q_values_2: torch.tensor with shape = (batch_size, num_actions)
            The Q-values that the current target_network estimates for taking action (a') from then next state
            (s') for each example in the batch (i.e. Q(s', a') for all a').
        :param actions: torch.tensor of shape = (batch_size, )
            The actions that the RL agent actually took from the current state (i.e. a)
        :param rewards: torch.tensor of shape = (batch_size, )
            The rewards that the RL agent recieved after taking action a from state s.
        :param terminated_mask: torch.tensor with shape = (batch_size,)
            A boolean mask of examples where the terminal state was reached and no more obs follow.
        :param truncated_mask: torch.tensor with shape = (batch_size,)
            A boolean mask of examples where the episode was truncated.
        :param wts: torch.tensor with shape = (batch_size, )
            A weight vector for compute the MSE that is returned by the replay buffer sampling method to
            un-bias the gradient update.
        :return: A torch.float giving the MSE loss computed over all examples in the batch.
        """
        gamma = self.config["hyper_params"]["gamma"]  # Get the temporal discount factor

        # Compute the argmax action in the next state s' by taking the max along dim=1 i.e. across all actions
        # for each example, which gives us a max_vals and argmaxes tensor of size (batch_size, )
        if self.config["model_training"]["loss"] == "double_dqn":  # If using the double DQN loss function,
            # then we use the q_network to select the argmax action (a') in the next state s'
            _, argmaxes = torch.max(q_values_2, dim=1)  # (batch_size, ) use the q_network to pick an action
            max_vals = target_q_values_2.gather(1, argmaxes.unsqueeze(1)).squeeze(1)  # (batch_size, )
            # Use the target_network to bootstrap and evaluate the best action in the next state
        else:  # Otherwise if using the standard DQN loss, we use the target network to select and evaluate
            # the choice in the next state s'
            max_vals, _ = torch.max(target_q_values_2, dim=1)

        # Compute the Q_samp values, zero out the second term if an obs is terminated (but not if truncated)
        Q_samp = rewards + (gamma * max_vals) * (~terminated_mask)

        # q_values (our y_hats) are (batch_size, num_actions), extract out the relevant entries by extracting
        # Q(s, a) for each action taken for each example in the batch i.e. select the appropriate col for each
        # row in q_values to make the relevant comparison. We have to unsqueeze to make the index tensor
        # the same dimensions as q_values, repeat the same action value "a" across each row
        Q_sa = torch.gather(q_values_1, dim=1, index=actions.long().unsqueeze(1)).squeeze(1)  # (batch_size, )
        td_errors = (Q_samp - Q_sa).abs()
        loss = (wts * td_errors.pow(2)).mean()  # Compute the weighted MSE loss function for all batch obs
        # After computing the loss, we should detach the td_errors from the gradient tracking computational
        # graph so that when we make updates to the replay buffer, gradients aren't being tracked there
        return loss, td_errors.detach().cpu()  # (torch.float, torch.Tensor of size (batch_size, ))

    def train(self, exp_schedule: LinearExploration, lr_schedule: LinearSchedule,
              beta_schedule: LinearSchedule) -> None:
        """
        Runs a full training loop to train the parameters of self.q_network using the reply buffer and fixed
        q-targets in self.target_network.

        The epsilon-greedy exploration is gradually decayed over time and controlled by exp_schedule.
        The learning rate is gradually decayed over time and controlled by lr_schedule.

        :param exp_schedule: A LinearExploration instance where exp_schedule.get_action(best_action) return
            an action that is either A). randomly selected or B). best_action and controlled by the internal
            epsilon parameter value.
        :param lr_schedule: A schedule for the learning rate where lr_schedule.param tracks it over time.
        :param beta_schedule: A schedule for the beta used in replay buffere weighted sampling.
        :return: None. Model weights and outputs are saved to disk periodically and also at the end.
        """
        self.logger.info(f"Training model: {self.config['model']}")
        self.logger.info(f"Running model training on device: {self.device}")

        # 0). Check that the q_network and target_network are initialized and copy the params from Q to tgt
        for x in ["q_network", "target_network", "optimizer"]:
            assert getattr(self, x) is not None, f"{x} is not initialized"
        self.update_target_network()  # Copy over the parameters to sync them

        # 1). Initialize the replay buffer and associated variables, it will keep track of recent obs so that
        #     we can maximize the amount of training we can get from them and it will also stack frames
        replay_buffer = ReplayBuffer(size=self.config["hyper_params"]["buffer_size"],
                                     frame_hist_len=self.config["hyper_params"]["state_history"],
                                     device=self.device, max_val=self.config["env"]["max_val"],
                                     eps=self.config["hyper_params"]["eps"],
                                     alpha=self.config["hyper_params"]["alpha"],
                                     seed=self.config["env"].get("seed"))

        # 2). Collect recent rewards and q-values in deque data structures and init other tracking vars
        # Track the rewards after running each episode to completion or truncation / termination
        episode_rewards = deque(maxlen=self.config["model_training"]["num_episodes_test"])
        max_q_values = deque(maxlen=1000)  # Track the recent max_q_values we get from the q_network
        q_values = deque(maxlen=1000)  # Track the recent q_values from the q_network across all actions
        self.init_averages()  # Used for tracking progress via Tensorboard

        t, last_eval, last_record = 0, 0, 0  # These counter vars are used by triggers
        # t = tracks the global number of timesteps so far i.e. how many time we call self.env.step(action)
        # last_eval = records the value of t at which we last ran an self.evaluation()
        # last_record = records the value of t at which we ran self.record()

        # Record one episode at the beginning before training if set to True in the config
        if self.config["env"].get("record", None):  # Record must be specified and set to True
            self.record(t)

        # Compile a list of evaluation scores, begin with an eval score run with the model's current weights
        eval_scores = [(t, self.evaluate()), ]  # List of scores computed for each evaluation run

        prog = Progbar(target=self.config["hyper_params"]["nsteps_train"])  # Training progress bar

        # 3). Interact with the environment, take actions, get obs + rewards, and update network params
        while t < self.config["hyper_params"]["nsteps_train"]:  # Loop until we reach the global training
            # step limit across all episodes, keep running episodes through the env until the limit is reached

            episode_reward = 0  # Track the total reward from all actions during the episode
            state = self.env.reset()  # Reset the env to begin a new training episode
            # replay_buffer.add_entry(state, 0, 0, 0, 0)  # Add a new entry to the replay buffer using the
            # initial frame from the env, we record (s', a, r, terminated, truncated) tuples

            while True:  # Run an episode of obs -> action -> obs -> action in the env until finished which
                # happens when either 1). the episode has been terminated by the env 2). the episode has
                # been truncated by the env or 3). the total training steps taken exceeds nsteps_train

                t += 1  # Increment the global training step counter i.e. every step of every episode +1

                # Decay the exploration rate, learning rate and beta as we go, update them for the current t
                exp_schedule.update(t)
                lr_schedule.update(t)
                beta_schedule.update(t)

                q_network_input = replay_buffer.get_stacked_obs()  # Get the most recent state obs that is
                # frame stacked if there is one to grab from the replay buffer, q_network_input=None
                # (batch_size=1, frame_stack, img_h, img_w, img_c) e.g. [1, 4, 80, 80, 1])

                # Choose and action according to current Q Network and exploration parameter epsilon, pass
                # in the stacked frame set to the q_network and generate a best action, get_best_action does
                # not track gradients since we're just generating data in the env rather than computing grads
                best_action, q_vals = self.get_best_action(q_network_input, default=0)
                action = exp_schedule.get_action(best_action)

                # Store the q values from the learned q_network in the deque data structures
                q_vals = q_vals.squeeze(0).cpu().numpy()  # Convert to numpy, for tracking purposes
                max_q_values.append(np.max(q_vals))  # Keep track of the max q-value returned by the q_network
                q_values.append(np.mean(q_vals))  # Keep track of the avg q-value returned bt the q_network

                # Perform the selected action in the env, get the new state, reward, and stopping flags
                new_state, reward, terminated, truncated, info = self.env.step(action)

                # Record the (s', a, r, terminated, truncated, t) transition in the replay buffer, the prior
                # state s is implicitly recorded by whatever was recorded in the replay buffer immediately
                # prior to the current write. Note that state here is the game screen img that has already
                # been down-sampled, max-pooled, reshaped and converted to grayscale images (80 x 80 x 1) and
                # are np.ndarrays of int type.
                replay_buffer.add_entry(new_state, action, reward, terminated, truncated)
                reward = np.clip(reward, -1, 1)  # We expect +/-1, but add reward clipping for stability

                # Track the total reward throughout the full episode
                episode_reward += reward

                # Perform a training step using the replay buffer to update the network parameters
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.param,
                                                       beta_schedule.param)

                if t % self.config["model_training"]["log_freq"] == 0:  # Update logging every so often
                    if t >= self.config["hyper_params"]["learning_start"]:
                        # Wait until the warm-up period has been reached to start logging
                        self.update_averages(episode_rewards, max_q_values, q_values, eval_scores)
                        self.add_summary(loss_eval, grad_eval, t)
                        if len(episode_rewards) > 0:  # If we have run at least 1 episode so far
                            prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg_R", self.avg_reward),
                                                      ("Max_R", np.max(episode_rewards)),
                                                      ("eps", exp_schedule.param),
                                                      ("Grads", grad_eval), ("Max_Q", self.max_q),
                                                      ("lr", lr_schedule.param)],
                                        base=self.config["hyper_params"]["learning_freq"])

                    else:  # If t < self.config["hyper_params"]["learning_start"], within the warm-up period
                        learning_start = self.config['hyper_params']['learning_start']
                        sys.stdout.write(f"\rPopulating the replay buffer {t}/{learning_start}...")
                        sys.stdout.flush()
                        prog.reset_start()

                # End the episode if one of the stopping conditions is met
                if terminated or truncated or t >= self.config["hyper_params"]["nsteps_train"]:
                    break

            # Perform updates at the end of each episode
            episode_rewards.append(episode_reward)  # Record the total reward received during the last episode

            if (t - last_eval) >= self.config["model_training"]["eval_freq"]:
                # If it has been more than eval_freq steps since the last time we ran an eval then run again
                if t >= self.config["hyper_params"]["learning_start"]:
                    last_eval = t  # Record the training timestemp of the last eval (now)
                    eval_scores.append((t, self.evaluate()))

            if self.config["env"].get("record", None):  # If the config says to periodically record
                if (t - last_record) >= self.config["model_training"]["record_freq"]:  # Hit record freq
                    if t >= self.config["hyper_params"]["learning_start"]:  # Past warm-up period
                        last_record = t  # Record the training timestep of the last record (now)
                        self.record(t)

        # Final screen updates
        self.logger.info("Training done.")
        self.save()  # Save the final model weights after training has finished
        eval_scores.append((t, self.evaluate()))  # Evaluate 1 more time at the end with the final weights
        save_eval_scores(eval_scores, self.config["output"]["plot_output"])

        # Record one episode at the end of training if set to True in the config
        if self.config["env"].get("record", None):
            self.record(t)

    def train_step(self, t: int, replay_buffer: ReplayBuffer, lr: float, beta: float) -> None:
        """
        Perform 1 training step to update the trainable network parameters of the self.q_network.

        :param t: The timestep of the current iteration.
        :param reply_buffer: A reply buffer used for sampling recent observations.
        :param lr: A float denoting the learning rate to use for this update.
        :param beta: A hyper-parameter used in prioritized experience replay sampling.
        :return: None.
        """
        loss_eval, grad_eval = 0, 0

        # Perform a training step parameter update
        learning_start = self.config["hyper_params"]["learning_start"]  # Warm-up period
        learning_freq = self.config["hyper_params"]["learning_freq"]  # Frequency of q_network param updates

        if t >= learning_start and t % learning_freq == 0:  # Update the q_network parameters with samples
            # from the reply buffer, which we only do every so often during game play
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr, beta)

        # Occasionally update the target network with the Q network parameters
        if t % self.config["hyper_params"]["target_update_freq"] == 0:
            self.update_target_network()

        # Occasionally save the model weights during training
        if t % self.config["model_training"]["saving_freq"] == 0:
            self.save()

        return loss_eval, grad_eval

    def update_step(self, t: int, replay_buffer: ReplayBuffer, lr: float, beta: float) -> Tuple[int, int]:
        """
        Performs an update of the self.q_network parameters by sampling from replay_buffer.

        :param t: The global training timestep counter (across all episodes and iterations).
        :param replay_buffer: A ReplayBuffer instance where .sample() gives us batches.
        :param lr: The learning rate to use when making gradient descent updates to self.q_network.
        :param beta: A hyper-parameter used in prioritized experience replay sampling.
        :return: The loss = (Q_samp(s) - Q(s, a))^2 and the total norm of the parameter gradients.
        """
        batch_size = self.config["hyper_params"]["batch_size"]

        # 1). Sample from the reply buffer to get recent (state, action, reward) values
        (state_batch, action_batch, reward_batch, next_state_batch,
         term_mask_batch, trunc_mask_batch, wts, indices) = replay_buffer.sample(batch_size, beta)
        # [state_batch, next_state_batch]: (batch_size, frame_hist, img_h, img_w, img_c)
        # [action_batch, reward_batch, term_mask_batch, trunc_mask_batch, wts, indices] -> (batch_size, )

        # 2). Check that required components are present
        msg = "WARNING: Networks not initialized. Check initialize_models"
        assert self.q_network is not None and self.target_network is not None, msg
        assert self.optimizer is not None, "WARNING: Optimizer not initialized. Check add_optimizer"

        # 3). Zero the tracked gradients of the q_network model
        self.optimizer.zero_grad()

        # 4). Run a forward pass of the batch of states through the q_network and generate Q-values for them
        # i.e. what the q_network estimates are the values of actions (a) from current states (s), Q(s, a)
        q_values_1 = self.get_q_values(state_batch, "q_network")

        # 5). Compute the q_values of state 2 using the q_network if needed i.e. if using double DQN
        if self.config["model_training"]["loss"] == "double_dqn":  # Also compute the q-values from the
            # q_network for the next state reached as well, which is required for the double DQN loss since
            # we will use the q_network to compute the next state argmax as well
            with torch.no_grad():  # We don't need to track gradients here bc the argmax breaks the
                # computational graph i.e. gradients are only tracked for Q(s, a), not for determining the
                # next action a' in double DQN
                q_values_2 = self.get_q_values(next_state_batch, "q_network")
        else:  # Otherwise, skip computing these since that will add more computation time if not needed
            q_values_2 = None

        with torch.no_grad():  # Also compute the q-values of the next states we reach after taking action a
            # from state s through the target_network, no gradient tracking needed here since we only update
            # the parameters of self.q_network
            target_q_values_2 = self.get_q_values(next_state_batch, 'target_network')

        # 6). Compute compute gradients wrt to the MSE Loss function
        loss, td_errors = self.calc_loss(q_values_1, q_values_2, target_q_values_2, action_batch,
                                         reward_batch, term_mask_batch, trunc_mask_batch, wts)
        replay_buffer.update_priorities(indices, td_errors)  # Update the priorities for the obs sampled
        loss.backward()  # Compute gradients wrt to the trainable parameters of self.q_network

        # Apply grad clipping before using the optimizer to take a step
        total_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),
                                                    self.config["model_training"]["clip_val"])

        # 7). Update parameters with the optimizer by taking a step
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()  # Perform a gradient descent update step for all parameters
        return loss.item(), total_norm.item()  # Return the loss and the norm of the gradients

    def evaluate(self, num_episodes: int = None, verbose: bool = True) -> float:
        """
        Runs a series of N (num_episodes) episodes using the current model parameters to evaluate the current
        parameter set. Returns the average return per episode.

        :param env: The training environment to use, (defaults to self.env).
        :param num_episodes: The number of episodes to run to compute an average per episode return.
        :return: The average per episode return over num_episodes.
        """
        if num_episodes is None:  # Override with the default from the config if not specified
            num_episodes = self.config["model_training"]["num_episodes_test"]

        if verbose:
            self.logger.info(f"\nEvaluating N={num_episodes} episodes...")

        # Use a replay buffer as a deque to track the last K frames, no sampling done here, keep the size of
        # the buffer limited to state_history so that we can save space, no need to dim large tensors
        replay_buffer = ReplayBuffer(self.config["hyper_params"]["state_history"],
                                     self.config["hyper_params"]["state_history"],
                                     device=self.device, max_val=self.config["env"]["max_val"],
                                     eps=self.config["hyper_params"]["eps"],
                                     alpha=self.config["hyper_params"]["alpha"],
                                     seed=self.config["env"].get("seed"))
        episode_rewards = []  # Keep track of the rewards for each eval episode run

        for i in range(num_episodes):  # Run N episodes to perform an evaluation call
            episode_reward = 0  # Track the total reward from all actions during the episode
            state = self.env.reset()  # Reset the env to begin a new training episode
            # replay_buffer.add_entry(state, 0, 0, 0, 0)  # Add a new entry to the replay buffer using the
            # initial frame from the env, we record (s', a, r, terminated, truncated) tuples

            while True:  # Iterate until we reach the end of the episode (terminated or truncated)
                q_network_input = replay_buffer.get_stacked_obs()  # Get the most recent state obs that is
                # frame stacked if there is one to grab from the replay buffer, q_network_input=None
                # (batch_size=1, frame_stack, img_h, img_w, img_c) e.g. [1, 4, 80, 80, 1])

                # No gradient updates made during the eval episodes, handled internally within get_best_action
                action = self.get_best_action(q_network_input)[0]  # Get best action recommended by the model

                # Perform the selected action in the env, get the new state and reward
                new_state, reward, terminated, truncated, info = self.env.step(action)
                reward = np.clip(reward, -1, 1)  # We expect +/-1, but add reward clipping for stability

                # Record the (s', a, r, terminated, truncated, t) transition in the replay buffer
                replay_buffer.add_entry(new_state, action, reward, terminated, truncated)

                # Track the total reward throughout the full episode
                episode_reward += reward

                if terminated or truncated:  # Check for episode stopping conditions
                    break

            # Perform updates at the end of each episode
            episode_rewards.append(episode_reward)

        # Compute summary statistics on the evaluation episodes computed
        avg_reward = np.mean(episode_rewards)
        sigma_reward = np.std(episode_rewards)

        if num_episodes > 1 and verbose:  # So long as the num of episodes was > 1 we will have data to report
            self.logger.info(f"Average reward: {avg_reward:04.2f} +/- {sigma_reward:04.2f}")

        return avg_reward

    def record(self, timestamp: str) -> None:
        """
        This method records a video for 1 episode using the model's current weights and saves it to disk.
        """
        self.logger.info(f"Recording episode at training timestep: {timestamp}")
        record_path = self.config["output"]["record_path"]
        os.makedirs(record_path, exist_ok=True)  # Make the save directory if not already there
        self.config["record_toggle"][0] = True  # Toggle recordings on
        self.evaluate(1, False)  # Run 1 episode through the eval method to generate a recording
        self.config["record_toggle"][0] = False  # Switch recordings off again when finished
        self.env.reset()  # Reset the env to end the episode and finish the recording
