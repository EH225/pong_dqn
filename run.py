"""
This module configures an env and trains the model specified by the --config command line argument. Results
are saved to the directories specified in the config file specified.

See: https://ale.farama.org/environments/pong/ for a description of the environment. Each time you score a
point, the reward is +1, each time your opponent scores, the reward is -1. Each episode of 1 game to 21.
"""

import sys, os
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

import argparse, warnings
import gymnasium as gym
import ale_py # This is the env that supports the pong emulator
import utils
from utils.general import read_yaml, pong_img_transform, video_post_processing
from utils.wrappers import FrameSkipEnv
from core.base_components import LinearSchedule, LinearExploration
from core.torch_models import LinearDQN, NatureDQN

gym.register_envs(ale_py)

# Supress warnings from gym
warnings.filterwarnings("ignore", module=r"gym")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training, evaluation, and record loops for Pong DQN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config",
                        help="The name of the config file in the config dir to be used for model training.",
                        default="NatureDQN_debug")

    args = parser.parse_args()

    # Read in the config file specified by the user to be used for model training
    config = read_yaml(f"config/{args.config}.yml")

    # 1). Configure the environment using gym with the config file specified
    if config["env"]["env_name"] == "ALE/Pong-v5": # Training on the Atari Pong env

        # frameskip=4 means that we down-sample temporally by a factor 4 i.e. the agent picks an action, and
        # then that action is repeated for the next 4 frames in the ALE (Atari) emulator. This helps reduce
        # the number of game frames that need to be processed, pong runs at 60 frames per second and follows
        # the Nature DQN paper. This measure reduces the computational load during training and allows the
        # agent to play through episodes faster. Since we will implement this frame-skip behavior with max
        # pooling manually below, we will set it to be 1 here so that there is no frame skipping done by gym
        env = gym.make(config["env"]["env_name"], frameskip=1, full_action_space=False,
                       render_mode=config["env"]["render_mode"])

        # Add a wrapper from the gym built-ins to generate video recordings of episodes on demand.
        # Create a mutable data structure that we can edit on-the-fly to toggle recordings on / off
        print("record_path", config["output"]["record_path"])
        config["record_toggle"] = [False] # Mutable data-type that we can edit as needed
        env = gym.wrappers.RecordVideo(env, video_folder=config["output"]["record_path"],
                                       episode_trigger=lambda ep: config["record_toggle"][0])

        # Following the Nature DQN paper, we use frameskip=4 with max pooling to deal with flickering. Some
        # Atari sprites only appear on alternating frames, so we take the pixel-wise maximum over the last
        # two consecutive frames out of the batch of 4 and use that as 1 output frame. This “max” step ensures
        # that transient sprites don’t disappear in the input, making observations more stable.
        # This class also applies image pre-processing to the input state fed to the RL agent.
        # Game screen image frames come from the env as (210, 160, 3) RGB color images, we will instead
        # crop, down-sample, and cast to gray scale to reduce the dimensionality of the inputs
        env = FrameSkipEnv(env, skip=config["hyper_params"]["skip_frame"], preprocessing=pong_img_transform,
                           shape=(80, 80, 1), overwrite_render=config["env"]["overwrite_render"])

    elif config["env"]["env_name"] == "debug_test_env":  # Training on the debug test-env instead

        if config["model"] == "NatureDQN":
            env = utils.EnvTest(5, (80, 80, 3), config['env'].get('seed')) # (n_states, state_shape, seed)
        elif config["model"] == "LinearDQN":
            env = utils.EnvTest(5, (5, 5, 1), config['env'].get('seed')) # (n_states, state_shape, seed)
        else:
            raise ValueError(f"Model={config['model']} not recognized")

    else:
        sys.exit("Incorrectly specified env,  config['model'] should either be 'Pong-v5' or 'linear'.")

    # 2). Configure the exploration strategy with epsilon decay
    exp_schedule = LinearExploration(env, config["hyper_params"]["eps_begin"],
                                     config["hyper_params"]["eps_end"],
                                     config["hyper_params"]["eps_nsteps"])

    # 3) Configure the learning rate decay schedule
    lr_schedule = LinearSchedule(config["hyper_params"]["lr_begin"],
                                 config["hyper_params"]["lr_end"],
                                 config["hyper_params"]["lr_nsteps"])

    # 4). Configure the beta importance sampling bias correction increase schedule
    beta_schedule = LinearSchedule(config["hyper_params"]["beta_begin"],
                                   config["hyper_params"]["beta_end"],
                                   config["hyper_params"]["beta_nsteps"])

    # 4). Instantiate the model to be trained by providing the env and config
    if config["model"] == "NatureDQN":
        model = NatureDQN(env, config)
    elif config["model"] == "LinearDQN":
        model = LinearDQN(env, config)
    else:
        raise ValueError(f"Model={config['model']} not recognized")

    # 5). Train the model after configuring the env and schedulers (lr and epsilon)
    model.train(exp_schedule, lr_schedule, beta_schedule)

    # 6). Perform video recording post processing if applicable i.e. speed up the videos and cap length
    video_post_processing(config, time_ds=4, size_ds=1, max_len=30)

