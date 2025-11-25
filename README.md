# Atari Pong with Deep Reinforcement Learning Project
This repository contains code for the Atari Pong with Deep Reinforcement Learning using a CNN-based Deep Q-Network project. Below is a quick overview of the repo layout:
- `config`: This folder contains config files which specify the training parameters and output directories for model training.
- `core`: This folder contains the core components of the deep reinforcement models. It contains a general DQN class and torch models for the `q_network`, which sits on top of it. This directory contains the bulk of the code of this repo.
- `env_req`: This folder contains `environment.yml` and `environment_cude.yml` file denoting the configuration of the virtual environment required to run the modules of this project.
- `refs`: This folder contains reference materials (i.e. academic papers) discussing the deep reinforcement learning techniques used in this project.
- `results`: This folder holds all the results saved to disk during training including eval score plots, model weights, episode recordings etc.
- `utils`: This folder contains additional modules that act as utility functions to the code contained in `core/`.
- `run.py`: This is the main driver script of the project and is used to train the RL models i.e. `python run.py --config={config_name}`.

This project was based on the repo provided by assignment 2 of Stanford University's Reinforcement Learning (XCS234) course, with heavy modifications made to: 
- Improve the overall code structure, clarity, simplicity, documentation, and format
- Address potential issues in the way the original repo was configured
- Improve runtime efficiency by switching to a `torch` based replay buffer
- Add support for a double DQN loss function
- Add support for prioritized experience replay (PER)
- Add support for periodically recording episodes of gameplay
