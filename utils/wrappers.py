"""
This module contains wrapper classes that layer on top of the default Pong env to alter the behavior of the
env.step function to apply temporal down-sampling on the frames passed to the RL agent along with other image
pre-processing steps to convert the images from (210, 160, 3) -> (80, 80, 1) grayscale.
"""

import numpy as np
import gymnasium as gym
from utils.viewer import SimpleImageViewer
from collections import deque
from typing import Callable, Tuple


class FrameSkipEnv(gym.Wrapper):
    """
    This wrapper object class modifies an original env object to perform temporal down-sampling i.e. instead
    of feeding in every single frame from the env into our model, we perform the same action for 4 consecutive
    frames and use a max-pooling operation over the last 2 to create a final output frame, which is then fed
    into the RL model to generate the next action.

    This measure reduces the computational load during training and allows the agent to play through episodes
    faster. The max pooling over the last 2 frames of the set of frames skipped is used to deal with issues
    related to flickering. Some Atari sprites only appear on alternating frames, so we take the pixel-wise
    maximum over the last two consecutive frames out of the batch of 4 and use that as 1 output frame. This
    “max” step ensures that transient sprites don’t disappear in the input, making observations more stable.

    During each length n frame interval, we play the same action in the env, aggregate the rewards recieved
    across all frame iterations, and compute a new input state for the next call to model.get_action(state)
    by max-pooling over the last 2 frames in the interval. We also apply image pre-processing where we down
    sample the original input RGB frames of size (210, 160, 3) to shape = (80, 80, 1) gray-scale images to
    reduce the dimensionality of the input.
    """
    def __init__(self, env=None, skip: int = 4, preprocessing: Callable = None,
                 shape: Tuple[int] = (80, 80, 1), overwrite_render: bool = True, high: int = 255):
        """
        Return only every nth frame - downsample along the temporal dimension so that we don't have to
        process so many frames and can play more games faster. Also performs max-pooling along the last
        2 of the skipped frame set and applies image pre-processing.
        """
        super(FrameSkipEnv, self).__init__(env)
        # Maintain a deque for the recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2) # Used for pooling over the last 2 frames in the interval
        self._skip = skip # Record the frame skip frequency

        self.overwrite_render = overwrite_render # If True, then the _render() method of the env is overwriten
        # to visualize the effect of using the preprocessing function provided
        self.viewer = None
        self.preprocessing = preprocessing # Store the pre-processing function
        self.observation_space = gym.spaces.Box(low=0, high=high, shape=shape, dtype=np.uint8)
        self.high = high # Store the max value of the state (i.e. pixel intensity), usually 255


    def step(self, action: int) -> tuple:
        """
        Given an input action (a), take that same action over the next self.skip frames.
        Aggregate the rewards we get during those frames and apply max pooling over the obs_buffer
        holding obs (next state) from each frame.

        :param action: An input action selected by the RL agent to be played over the next self.skip frames.
        :returns: An aggregated set of outputs from the set of frames gone by in the self.skip interval:
            max_frame: A max-pooling over the last 2 steps i.e. max pooling over the last 2 game screen images
                which represents the state of the environment.
            total_reward: The total reward obtained during the set of frames
            terminated: Whether the current episode of the game is now terminated or not.
            truncated: Whether the current episode of the game has reached a truncation limit or not.
            info: Whatever info is returned by self.env.step(action) on the last frame of the interval.
        """
        total_reward = 0.0
        for _ in range(self._skip): # Iterate over this interval of frames
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs) # Keep track of the trailing frames so we can max pool at the end
            total_reward += reward # Aggregate the total rewards obtained during these frames
            if terminated or truncated: # Stop early if the episode was terminated or hits a truncation limit
                break

        max_pool_frame = np.max(np.stack(self._obs_buffer), axis=0) # Compute max pooling over the last 2
        # frames to mitigate issues related to flickering in the game screen display, this will be the 1
        # state we return and feed to our model for the next action generation

        # Return the same tuple of information as self.env.step(action) but aggregated throughout the
        # set of self.skip steps that have elapsed, apply the pre-processing function to the output state
        return self.preprocessing(max_pool_frame), total_reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        """
        Clears past frame buffer and re-initializes to first observation from inner env. This method is used
        to clear cached data between eposides so that we can start a new one with a fresh initialization.

        Returns the first state observation from the env after resetting.
        """
        self._obs_buffer.clear() # Clear the recent observation buffer (s') of game board images
        obs, info = self.env.reset() # Reset the env and obtain the starting state from the env
        self._obs_buffer.append(obs) # Add the starting state t othe obs buffer as the first state obs
        return self.preprocessing(obs)

    def _render(self, mode="human", close=False):
        """
        If  self.overwrite_render is True, then the _render() method of the env is overwriten to visualize the
        effect of using the preprocessing function provided.
        """

        if self.overwrite_render: # Then overwrite the render() method of the env
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return

            if mode == "rgb_array":
                return self.obs

            elif mode == "human":
                from gymnasium.envs.classic_control import rendering

                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                self.viewer.imshow(self.obs)

        else: # Otherwise call the _render method as per usual from the inherited parent class
            super(PreProcessingEnv, self)._render(mode, close)


### TODO: Depreciated and no longer used
class PreProcessingEnv(gym.Wrapper):
    """
    This wrapper object class modifies the input env by re-defining the step(), reset(), and _render() methods
    to accomidate a pre-processing callable function being applied to all obs (states) i.e. game board imgs.

    This function also overrides and redefines the _render() method potentially to visualize the effect of
    using the preprocessing i.e. instead of rendering videos of the game env using the original obs game
    frames, we can instead show it using the obs after pre-processing.
    """

    def __init__(self, env, preprocessing: Callable, shape: Tuple[int], overwrite_render: bool = True,
                 high: int = 255):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
            shape: (list) shape of obs after prepro
            overwrite_render: (bool) if True, render is overwriten to vizualise effect of prepro
            grey_scale: (bool) if True, assume grey scale, else black and white
            high: (int) max value of state after prepro
        """
        super(PreProcessingEnv, self).__init__(env)
        self.overwrite_render = overwrite_render # If True, then the _render() method of the env is overwriten
        # to visualize the effect of using the preprocessing function provided
        self.viewer = None
        self.preprocessing = preprocessing # Store the pre-processing function
        self.observation_space = gym.spaces.Box(low=0, high=high, shape=shape, dtype=np.uint8)
        self.high = high # Store the max value of the state (i.e. pixel intensity), usually 255

    def step(self, action: int) -> tuple:
        """
        This method overwrites the env.set() method to apply preprocessing to the states before passing them
        to the RL agent for processing i.e. to generate and selected action.

        :param action: An input action selected by the RL agent to be played over the next self.skip frames.
        :returns: The same outputs as self.env.step(action) with the output obs passed through the
            pre-processing function before being returned.
        """
        obs, reward, terminated, truncated, info = self.env.step(action) # Call step on env
        self.obs = self.preprocessing(obs) # Apply pre-processing function to next state (obs)
        return self.obs, reward, terminated, truncated, info # Return step outputs

    def reset(self, seed=None, options=None):
        """
        This method is used to clear cached data between eposides so that we can start a new one with a fresh
        initialization.

        Returns the first state observation from the env after resetting after pre-processing.
        """
        self.obs = self.preprocessing(self.env.reset())
        return self.obs

    def _render(self, mode="human", close=False):
        """
        If  self.overwrite_render is True, then the _render() method of the env is overwriten to visualize the
        effect of using the preprocessing function provided.
        """

        if self.overwrite_render: # Then overwrite the render() method of the env
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return

            if mode == "rgb_array":
                return self.obs

            elif mode == "human":
                from gymnasium.envs.classic_control import rendering

                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                self.viewer.imshow(self.obs)

        else: # Otherwise call the _render method as per usual from the inherited parent class
            super(PreProcessingEnv, self)._render(mode, close)
