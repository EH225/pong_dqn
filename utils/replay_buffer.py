import numpy as np
import torch

from typing import Tuple, Optional

class ReplayBuffer:
    """
    A memory-efficient implimentation of a replay buffer that accepts input np.ndarrays and outputs data as
    torch.tensor.

    A replay buffer acts as a recent memory bank for (s, a, r, s`) tuples observed by the RL agent as it
    interactions with the environment. By caching these values, we can randomly sample from them to generate
    low-correlation samples during gradient-based training updates to the model parameters that are not too
    heavily anchored toward recent env experiences.

    A replay buffer (also called experience replay) is a core component of training that significantly
    improves the learning stability and efficiency of a DQN model by reducing the correlation between
    consecutive samples, improves data efficiency by reusing obs from the env multiple times, smooths out the
    learning process, enables off-policy learning, and helps to avoid feedback loops during training.

    Since the DQN models are implemented using pytorch, this data structure will internally maintain values
    as torch.tensors and will return sampled data as torch.tensors as well. This is done to avoid time
    consuming transfers between numpy and torch, which can trigger data movements between the CPU and GPU.

    This implementation also allows for Prioritized Experience Replay (PER). PER assigns a priority to each
    transition—typically based on the TD error (the difference between predicted and target Q-values), and
    therefore samples the high-error transitions more often to help the RL agent learn faster by focusing on
    experiences where it struggled the most. This can help improve the speed of training and also the quality
    of the final trained model. Certain rate, but infrequent observations can be overshadowed by abundant,
    but uninformative transitions.
    """
    def __init__(self, size: int, frame_hist_len: int, device: str = "cpu", max_val: int = 255,
                 seed: Optional[int] = None):
        """
        Initialize the replay buffer with a max capacity of size where each observation sampled should consist
        of frame_hist_len historical observations. Data is stored on the device specified to match the
        location of the models that will utilize it for training.

        :param size: The max number of transition observations (s, a, r, s') recorded in the buffer. When the
            buffer size is full and a new entry is added, the oldest observations are overwritten, FIFO.
        :param frame_hist_len: The number of state observations returned for each sampled transition i.e. for
            the current state s, we return the last frame_hist_len state values ending at s.
        :param device: The device to store the data on as torch.tensors so that it is quickly utilizable by
            the models being trained.
        :param max_val: The max integer pixel value we expect to see in the input frames being saved to the
            buffer. We expect the inputs to be int values [0, max_val] which will be cast to float [0, 1].
        """
        self.size = size # Record the max capacity of the replay buffer
        self.frame_hist_len = frame_hist_len # Record how many frames to return for each state obs, this
        # allows for frame stacking i.e. feed in the last 4 game screen img obs to the model at time t
        self.device = device # Record the device where the data will be stored
        self.max_val = max_val # Record the max pixel intensity value we expect from the input frames which
        # is used to re-scale from int [0, max_val=255] to float [0, 1].

        self.last_idx = None # Track the last index at at which a new frame was written
        self.next_idx = 0 # Tracks the next index to write historical observational data to
        self.num_in_buffer = 0 # Tracks how many observations are currently saved in the buffer, <= size
        self.buffer_full = False # Set to True when the buffer reaches full capacity

        # Init variables to store the info for each transition observation
        self.obs = None # States (s) e.g. game screen frames
        self.actions = None # The action (a) taken from each starting state (s)
        self.rewards = None # The reward (r) recieved after taking action (a) from state (s)
        self.terminated = None # A bool indicating whether the episode has terminated
        self.truncated = None # A bool indicating whether the episode has been truncated
        # terminated signifies an episode ending due to a natural, task-specific condition like reaching a
        # goal state or failing. truncated means it ended because of an external constraint, such as a time
        # or step limit and requires bootstrapping the value function since the episode hasn't yet terminated

        self.rng = np.random.default_rng(seed) # Create a random number generator for sampling with a seed

    def _get_next_idx(self, idx: int) -> int:
        """
        Given an input idx, this function returns the next index with wrap around i.e. at idx == size - 1,
        the next value index is 0.

        :param idx: An input index in the range [0, self.size-1].
        :return: The next sequential index which is either idx + 1 or 0.
        """
        return (idx + 1) % self.size

    def add_entry(self, frame: np.ndarray, action: int, reward: float, terminated: bool,
                  truncated: bool) -> int:
        """
        Stores a single frame in the replay buffer at the next available index, overwriting old frames if
        necessary using a FIFO management system.

        :param frame: An input frame of size (img_h, img_w, img_c) and dtype np.uint8 to be stored.
        :param action: A non-negative integer denoting the action that was performed from this state (obs).
        :param reward: The reward received when the action was performed from this state.
        :param terminated: True if episode was terminated after performing that action.
        :param truncated: True if episode was truncated after performing that action.
        :return: An integer index designating the location where the frame was stored internally.
        """
        if self.obs is None:  # Auto-init the storage space based on the first frame saved to the buffer
            # obs holds game screen image frames and is of size (size, img_h, img_w, img_c)
            self.obs = torch.empty([self.size] + list(frame.shape), dtype=torch.float32, device=self.device)
            self.actions = torch.empty([self.size], dtype=torch.uint8, device=self.device) # (size, ) > 0
            self.rewards = torch.empty([self.size], dtype=torch.float32, device=self.device) # (size, )
            self.terminated = torch.empty([self.size], dtype=bool,device=self.device) # (size, ) bool values
            self.truncated = torch.empty([self.size], dtype=bool,device=self.device) # (size, ) bool values
        # Else we assume that these objects have already been instantiated

        # Store the values passed in the replay buffer at the next write location
        # Convert the input frame pixels from int[0, max_val] to float [0, 1] instead
        self.obs[self.next_idx] = torch.as_tensor(frame, device=self.device) / self.max_val
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.terminated[self.next_idx] = terminated
        self.truncated[self.next_idx] = truncated

        self.last_idx = self.next_idx # Record the index where this new frame was written to
        self.next_idx = self._get_next_idx(self.next_idx) # Update next_idx to the next write location, wrap
        # around back to 0 at the beginning again if we reach the end
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1) # Update the number of elements in the
        # buffer which is capped at self.size, but is generally 1 larger than before
        self.buffer_full = self.num_in_buffer == self.size # Track if the buffer has been fully filled
        return self.last_idx # Return the index in the replay buffer where this frame was stored

    def get_stacked_obs(self, idx: int = None) -> torch.tensor:
        """
        Returns a frame stacked torch.tensor of size (frame_hist_len, img_h, img_w, img_c) which is a state
        observation for an RL agent model i.e. the model expects as input a stacked tensor of the last
        K frames.

        :param idx: The index of the last frame of the state obs within the replay buffer. If left as None,
            then idx will default to self.last_idx i.e. the last frame input into the buffer.
        :returns: A torch.tensor of size (batch_size=1, frame_hist_len, img_h, img_w, img_c) representing a
            state obs with a batch_size of 1.
        """
        idx = idx if idx is not None else self.last_idx # Use the idx most recently written to if None
        bool_1 = idx is not None

        if bool_1 and idx < self.num_in_buffer and (idx >= self.frame_hist_len - 1 or self.buffer_full):
            # Check that idx is not None i.e. self.last_idx is not none and we have written to the buffer
            # Also check that the idx is not beyond the last buffer element i.e. not in the zeros region
            # Also check that the idx requested has sufficiently many prior frames to stack with it
            # If all of those conditions are met, then return a stacked set of frames with idx being the last
            # one in the stack along with the (self.frame_hist_len - 1) prior frames
            start_idx = (idx - self.frame_hist_len + 1) % self.size # Find the index of the first frame
            # Look at the frames we will include in this stacked context, say we have indices 0, 1, 2, 3
            # and at index 1 truncated or terminated is True, then the first 2 frames should be blank since
            # the frame at index 2 is at the start of a new episode. idx 0 and 1 do not preceed idx 2's state
            i = start_idx
            for _ in range(self.frame_hist_len - 1): # Iterate over all but the last one
                next_i = self._get_next_idx(i)
                if self.truncated[i] or self.terminated[i]:
                    start_idx = next_i # Move up the start idx to the next index since the episode ended
                i = next_i # Update for next iteration

            # Now extract the frames from the data structures we have internally
            if start_idx <= idx: # If the frames are contiguious, then pull them out directly
                state_obs = self.obs[start_idx:(idx + 1), :, :, :]
            else: # Otherwise we have a situation where we have wrap around and start_idx > idx
                state_obs = torch.concat([self.obs[start_idx:, ...], self.obs[:(idx + 1), ...]], axis=0)

            # However many preceeding frames are missing, we will pad with 0 frames at the start
            if state_obs.shape[0] < self.frame_hist_len:
                zero_padding_shape = [self.frame_hist_len - state_obs.shape[0]] + list(state_obs.shape[1:])
                state_obs = torch.concat([torch.zeros(zero_padding_shape), state_obs]) # Add zero frames
            return state_obs.unsqueeze(0) # (batch_size=1, frame_hist_len, img_h, img_w, img_c)


    def sample(self, batch_size: int) -> tuple:
        """
        This method randomly samples batch_size transition observations from the replay buffer which each
        describes a (s, a, r, s') tuple. If there are not sufficiently many observations in the replay
        buffer, then None is returend.

        :param batch_size: The number of randomly sampled historical examples to return from the buffer.
        :returns: Returns a tuple of data:
            obs_batch: (batch_size, frame_hist_len, img_h, img_w, img_c)
            action_batch: (batch_size, )
            reward_batch: (batch_size, )
            next_obs_batch: (batch_size, frame_hist_len, img_h, img_w, img_c)
            terminated_batch: (batch_size, )
            truncated_batch: (batch_size, )
        """
        assert isinstance(batch_size, int) and batch_size >= 1, "batch_size must be an int >= 1"
        msg = "batch_size exceeds number of examples in the buffer needed"
        assert batch_size <= min(self.size, self.num_in_buffer + self.frame_hist_len), msg
        # Case 1: There is no wrap around yet, then we can sample idx [frame_hist_len-1, num_samples-2]
        # In order to have frame_hist_len frames, frame_hist_len-1 is the smallest index we can use
        # The last frame saved to the buffer will be in idx=num_samples-1, but we also need a next state so
        # we will set an upper bound on the indices we can sample to be num_samples-2 so there is a next state
        indices = self.rng.choice(np.arange(self.frame_hist_len - 1, self.num_in_buffer - 1),
                                  size=batch_size, replace=False)

        # Case 2: The buffer has been filled at least once before so there is wrap around. Now the oldest
        # frame isn't at idx=0 but rather at self.next_idx which is somewhere in the middle of the array.
        # We can use the indices from above and shift them accordingly so that idx=0 from above corresponds
        # to self.next_idx instead, this will preserve the logic of the min and max idx values to sample
        if self.buffer_full and self.next_idx != 0:
            indices = (indices + self.next_idx) % self.size

        indices = torch.from_numpy(indices).to(self.device) # Move indices to the same device as the data

        ## TODO: Need to implement the priorty sampling here, need to record the error or something like that
        # As well, but for now for simplicity we can begin with the current simple sampling approach to check
        # that is still works

        obs_batch = torch.concatenate([self.get_state_obs(idx) for idx in indices], axis=0)
        action_batch = self.actions[indices] # The action that took ups from (s) to (s')
        reward_batch = self.rewards[indices] # The reward obtained from (s) to (s')
        terminated_batch = self.terminated[indices] # Whether s' is a terminated state
        truncated_batch = self.truncated[indices] # Whether s' is a truncated state

        state_shape = [1, self.frame_hist_len] + list(self.obs.shape[1:]) # The size of the zero padding
        next_obs_batch = torch.concatenate(
            [self.get_state_obs(self._get_next_idx(idx)) if not terminated_batch[i] or truncated_batch[i]
             else torch.zeros(state_shape) for i, idx in enumerate(indices)], axis=0)

        return obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, truncated_batch

### Perhaps we should do something that is better at avoiding bad edge cases. Should probably do some kind of
### valid indicy stuff
### Need to handle things here more carefully and make sure that we have the correct masking and such, review
### the hints from Chat GPT3


