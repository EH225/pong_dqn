import numpy as np
import torch
from typing import Tuple, Optional


class ReplayBuffer:
    """
    A memory-efficient implementation of a replay buffer that accepts input np.ndarrays and outputs data as
    torch.tensor.

    A replay buffer acts as a recent memory bank for (s', a, r, term, trun) tuples observed by the RL agent as
    it interacts with the environment. By caching these values, we can randomly sample from them to generate
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
                 eps: float = 1e-5, alpha: float = 0.6, seed: Optional[int] = None):
        """
        Initialize the replay buffer with a max capacity of size where each observation sampled should consist
        of frame_hist_len historical observations. Data is stored on the device specified to match the
        location of the models that will utilize it for training.

        :param size: The max number of transition observations (s', a, r, term, trun) recorded in the buffer.
            When the buffer size is full and a new entry is added, the oldest obs are overwritten, FIFO.
        :param frame_hist_len: The number of state observations returned for each sampled transition i.e. for
            the current state s, we return the last frame_hist_len frames ending at s.
        :param device: The device to store the data on as torch.tensors so that it is quickly utilizable by
            the models being trained.
        :param max_val: The max integer pixel value we expect to see in the input frames being saved to the
            buffer. We expect the inputs to be int values [0, max_val] which will be cast to float [0, 1].
        :param eps: The epsilon value to use when updating priority values.
        :param alpha: Used to control the degree of priority sampling. When alpha=0, then no priority
            sampling is performed, all indices have an equal change of being sampled. When alpha=1, then we
            have maximal prioritization of the larger TD error obs.
        :param seed: An optional seed that can be set which controls the selection of random samples.
        """
        assert size >= frame_hist_len, "size must be >= frame_hist_len else stacked obs cannot be returned"
        self.size = int(size)  # Record the max capacity of the replay buffer
        self.frame_hist_len = int(frame_hist_len)  # Record how many frames to return for each state obs, this
        # allows for frame stacking i.e. feed in the last 4 game screen img obs to the model at time t
        self.device = device  # Record the device where the data will be stored
        self.max_val = max_val  # Record the max pixel intensity value we expect from the input frames which
        # is used to re-scale from int [0, max_val=255] to float [0, 1].

        self.last_idx = None  # Track the last index at which a new frame (s') was written
        self.next_idx = 0  # Tracks the next index to write historical observational data to
        self.num_in_buffer = 0  # Tracks how many observations are currently saved in the buffer, <= size
        self.buffer_full = False  # Set to True when the buffer reaches full capacity

        # Init variables to store the info for each transition observation
        self.frames = None  # Next states (s') e.g. game screen frames we transition to after the action (a)
        self.actions = None  # The action (a) taken from each starting state (s)
        self.rewards = None  # The reward (r) received after taking action (a) from state (s)
        self.terminated = None  # A bool indicating whether the episode has terminated at frame (s')
        self.truncated = None  # A bool indicating whether the episode has been truncated at frame (s')
        # terminated signifies an episode ending due to a natural, task-specific condition like reaching a
        # goal state or failing. truncated means it ended because of an external constraint, such as a time
        # or step limit and requires bootstrapping the value function since the episode hasn't yet terminated

        self.priority = None  # The |TD error| + eps, used in sampling for prioritized experience replay
        self.eps = float(eps)  # Epsilon value for computing priority scores i.e. p = (td_err + eps) ** alpha
        self.alpha = float(
            alpha)  # Controls how much more we sample the high priority obs, alpha=0 equal prob
        self.max_priority = float(eps)  # The priority values are all initialized at eps

        self.rng = np.random.default_rng(seed)  # Create a random number generator for sampling with a seed

    def _get_next_idx(self, idx: int) -> int:
        """
        Given an input idx, this function returns the next index with wrap around i.e. at idx == size - 1,
        the next value index is 0. This is the inverse function of _get_prior_idx.

        :param idx: An input index in the range [0, self.size-1].
        :return: The next sequential index which is either idx + 1 or 0.
        """
        return (idx + 1) % self.size

    def _get_prior_idx(self, idx: int) -> int:
        """
        Given an input idx, this function returns the prior index with wrap around i.e. 1 -> 0 and at the
        start we get 0 -> size - 1. This is the inverse function of _get_next_idx.

        :param idx: An input index in the range [0, self.size - 1].
        :return The prior sequential index which is either idx - 1 or self.size - 1.
        """
        return (idx - 1) % self.size

    def add_entry(self, frame: np.ndarray, action: int, reward: float, terminated: bool,
                  truncated: bool) -> int:
        """
        Stores a single frame in the replay buffer at the next available index, overwriting old frames if
        necessary using a FIFO management system. Given the following interaction in the env (s, a, r, s'),
        this method stores the data of (s', a, r, terminated, truncated). Prior states are implicitly inferred
        at the frames written to prior indices within the buffer.

        :param frame: An input frame of size (img_h, img_w, img_c) and dtype np.uint8 to be stored.
        :param action: A non-negative integer denoting the action that was performed from the prior state.
        :param reward: The reward received when the action was performed from the prior state.
        :param terminated: True if the episode was terminated upon reaching the current state s'.
        :param truncated: True if the episode was truncated upon reaching the current state s'.
        :return: An integer index designating the location where the frame was stored internally.
        """
        if self.frames is None:  # Auto-init the storage space based on the first frame saved to the buffer
            # obs holds game screen image frames and is of size (size, img_h, img_w, img_c)
            self.frames = torch.zeros(tuple([self.size] + list(frame.shape)), dtype=torch.uint8,
                                      device=self.device)
            self.actions = torch.zeros((self.size,), dtype=torch.long, device=self.device)  # (size, ) > 0
            self.rewards = torch.zeros((self.size,), dtype=torch.float32, device=self.device)  # (size, )
            self.terminated = torch.zeros((self.size), dtype=torch.bool, device=self.device)  # (size, )
            self.truncated = torch.zeros((self.size,), dtype=torch.bool, device=self.device)  # (size, )
            self.priority = torch.full((self.size,), self.eps, dtype=torch.float32, device=self.device)
        # Else we assume that these objects have already been instantiated

        # Store the values passed in the replay buffer at the next write location
        self.frames[self.next_idx] = torch.as_tensor(frame, device=self.device)
        self.actions[self.next_idx] = int(action)
        self.rewards[self.next_idx] = float(reward)
        self.terminated[self.next_idx] = bool(terminated)
        self.truncated[self.next_idx] = bool(truncated)
        self.priority[self.next_idx] = self.max_priority  # Init as the max priority seen so far to make sure
        # that new obs have a high probability of being sampled at least 1x when they enter the buffer

        self.last_idx = self.next_idx  # Record the index where this new frame was written to
        self.next_idx = self._get_next_idx(self.next_idx)  # Update next_idx to the next write location, wrap
        # around back to 0 at the beginning again if we reach the end
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)  # Update the number of elements in the
        # buffer which is capped at self.size, but is generally 1 larger than before
        self.buffer_full = (self.num_in_buffer == self.size)  # Track if the buffer has been fully filled
        return self.last_idx  # Return the index in the replay buffer where this frame was stored

    def get_stacked_obs(self, idx: int = None, frame_hist_len: int = None) -> torch.tensor:
        """
        Returns a frame stacked torch.Tensor of size (batch_size=1, frame_hist_len, img_h, img_w, img_c),
        which is a state observation for an RL agent model i.e. the model expects as input a stacked tensor
        of the last K (usually 4) frames.

        :param idx: The index of the last frame of the state (s') within the replay buffer. If left as None,
            then idx will default to self.last_idx i.e. the last frame input into the buffer.
        :param frame_hist_len: The frame history length to use when stacking frames i.e. the frame lookback.
            If left as None, then this will default to the internal self.frame_hist_len parameter.
        :returns: A torch.tensor of size (batch_size=1, frame_hist_len, img_h, img_w, img_c) representing a
            state obsercation with a batch_size of 1.
        """
        assert self.last_idx is not None, "No frames yet written to buffer, nothing to return"
        idx = self.last_idx if idx is None else idx  # Default to last written frame if not specified
        assert 0 <= idx < self.num_in_buffer, f"idx={idx} out of range: [0, {self.num_in_buffer} - 1]"
        # We allow the frame_hist_len to be flexible so that this method can run for frame_hist_len + 1 too
        # which will make deriving s and s' during sampling faster and easier since they are overlapping
        frame_hist_len = self.frame_hist_len if frame_hist_len is None else frame_hist_len

        # 1). Determine the correct start_idx i.e. the first frame in the stack that is also part of this ep
        start_idx = (idx - frame_hist_len + 1) % self.size  # Get the index of the first frame to include
        # There are a number of possible issues we could run into using this start_idx, adjust it forward
        # to skip over any frames that are not part of the current episode, look out for truncation,
        # termination, and where the last_idx is located

        i = start_idx  # Begin at the original start_idx
        while i < idx:  # Iterate over all the indices from start_idx up to but not including idx which we
            # must return as the final frame in the stacked frame set
            if self.truncated[i] or self.terminated[i] or i == self.last_idx:
                # 1). If the ith frame is a terminal state, then it belongs to a different episode, increment
                # the start_idx counter beyond so that we do not include or any frames prior which will also
                # be part of a different episode i.e. will replace this frame with a zero frame instead
                # 2). Also if the ith frame is the last write location, then that means our idx frame which
                # comes after in memory will be temporally before the ith frame, so also zero it out
                start_idx = i + 1
            i += 1  # Move to the next index location to review
        # Now we have determined the correct start_idx which will be <= idx and will contain frames that are
        # all part of the current episode, any others we need we will prepend as zero frames. If the buffer
        # is not yet full and idx=0, it is okay for start_idx to be from the end which is empty

        # 2). Extract the relevant frames between start_idx -> idx
        if start_idx <= idx:  # If the frames are contiguous, then pull them out directly
            stacked_obs = self.frames[start_idx:(idx + 1), :, :, :]
        else:  # Otherwise we have a situation where we have wrap around and start_idx > idx
            stacked_obs = torch.concat([self.frames[start_idx:, ...], self.frames[:(idx + 1), ...]], axis=0)

        # 3). However, many preceding frames are missing, add 0 frames there to compensate
        if stacked_obs.shape[0] < frame_hist_len:
            zero_padding_shape = [frame_hist_len - stacked_obs.shape[0]] + list(stacked_obs.shape[1:])
            stacked_obs = torch.concat([torch.zeros(zero_padding_shape), stacked_obs])  # Add zero frames

        # Convert the frame pixels from int[0, max_val] to float [0, 1] instead
        return stacked_obs.unsqueeze(0) / self.max_val  # (batch_size=1, frame_hist_len, img_h, img_w, img_c)

    def sample(self, batch_size: int, beta: float = 0.1) -> Tuple[torch.tensor]:
        """
        This method randomly samples batch_size transition observations from the replay buffer which each
        describe a (s, a, r, s') interaction with the env. If there are no sufficiently many observations
        in the reply buffer, then an error is raised.

        :param batch_size: The number of randomly sampled historical examples to return from the buffer.
        :param beta: Because PER distors the real data distribution, we use importance sampling to un-bias
            our gradient updates which is controlled by this beta parameter which should be annealed from
            small to large during training.
        :returns: Returns a tuple of data:
            stacked_obs_batch: (batch_size, frame_hist_len, img_h, img_w, img_c)
            action_batch: (batch_size, )
            reward_batch: (batch_size, )
            next_stacked_obs_batch: (batch_size, frame_hist_len, img_h, img_w, img_c)
            terminated_batch: (batch_size, )
            truncated_batch: (batch_size, )
            indices: (batch_size, )
        """
        assert isinstance(batch_size, int) and batch_size >= 1, "batch_size must be an int >= 1"
        assert batch_size <= self.num_in_buffer, "batch_size exceeds number of examples in the buffer needed"
        # Randomly sample indices of s' to include in this batch in the range [0, num_in_batch - 1]
        if self.alpha > 0:  # Use priority sampling, when alpha == 0, then we're using uniform sampling
            probs = (self.priority[:self.num_in_buffer] + self.eps) ** self.alpha  # Compute the priority wts
            probs /= probs.sum()  # Normalized to be a probability vector
            indices = torch.multinomial(probs, batch_size, replacement=False)  # Take a weighted sample of idx
            wts = ((1 / (probs[indices] * batch_size)) ** beta)  # Extract the relevant weights
        else:  # Otherwise, use naive sampling where all indices have an equal change of being selected
            indices = self.rng.choice(np.arange(0, self.num_in_buffer), size=batch_size, replace=False)
            wts = 1 / torch.ones(batch_size, device=self.device)  # Equal 1/n weights for all batch elements
            indices = torch.from_numpy(indices).to(self.device)  # Move indices to the same device as the data

        # For each index selected, get the stacked obs with a length of frame_hist_len + 1 so that we have
        # both s and s' in the same tensor since they are overlapping and share the same middle frames
        _stacked_obs = [self.get_stacked_obs(int(idx), self.frame_hist_len + 1) for idx in indices]
        # Extract s and s' from each entry of _stacked_obs, the first dim is batch_size=1, slice along dim=1
        stacked_obs_batch = torch.concatenate([x[:, :-1] for x in _stacked_obs], axis=0)
        next_stacked_obs_batch = torch.concatenate([x[:, 1:] for x in _stacked_obs], axis=0)
        action_batch = self.actions[indices]  # The action that took ups from (s) to (s')
        reward_batch = self.rewards[indices]  # The reward obtained from (s) to (s')
        terminated_batch = self.terminated[indices]  # Whether s' is a terminated state
        truncated_batch = self.truncated[indices]  # Whether s' is a truncated state

        return (stacked_obs_batch, action_batch, reward_batch, next_stacked_obs_batch,
                terminated_batch, truncated_batch, wts, indices)

    def update_priorities(self, indices: torch.tensor, td_errors: torch.tensor) -> None:
        """
        Updates the priority scores associated with the indices provided, which determined how likely a given
        observation is to be sampled during training i.e. the larger the td_error, the more likely an obs is
        to be selected during sampling.

        :param indices: The indices associated with the td_errors provided.
        :param td_errors: A tensor of TD errors i.e. |Q(s, a) - (r + gamma * max(Q(s')))|
        :returns: None, modifies the internal data structure holding the priorities for each obs.
        """
        priorities = (td_errors + self.eps) ** self.alpha  # Compute updated priority values from TD diffs
        self.priority[indices] = priorities  # Update values internally
        self.max_priority = max(self.max_priority, priorities.max())  # Update the max globally priority

# if __name__ == "__main__":
#     replay_buffer = ReplayBuffer(10, 4, "cpu", 1)
#     for i in range(1, 15):
#         frame = np.ones((2, 2, 1)) * i
#         idx = replay_buffer.add_entry(frame, 0, 0, False, False)
#         print("Added at idx", idx)

#         res = replay_buffer.get_stacked_obs()
#         print("res.shape", res.shape)
#         print("res.sum()", res.sum())

#         print("stacked_obs")
#         for i in range(4):
#             print(res[0, i, :, :, 0])

#         print("buffer internal")
#         for i in range(10):
#             print(replay_buffer.frames[i, :, :, 0])

#         input("press enter to continue: ")

#     (stacked_obs_batch, action_batch, reward_batch, next_stacked_obs_batch,
#             terminated_batch, truncated_batch, indices) = replay_buffer.sample(3)

#     # Check that all the frames overlap where we expect them to in s and s'
#     assert (stacked_obs_batch[:, 1:, ...] == next_stacked_obs_batch[:, :-1, ...]).all()

#     assert action_batch.sum() == 0
#     assert reward_batch.sum() == 0
#     assert terminated_batch.sum() == 0
#     assert truncated_batch.sum() == 0

#     for i in range(next_stacked_obs_batch.shape[0]):
#         print("\n\nNext Image stack")
#         for j in range(next_stacked_obs_batch.shape[1]):
#             print(next_stacked_obs_batch[i, j, :, :, 0])


# buffer internal
# tensor([[11., 11.],
#         [11., 11.]])
# tensor([[12., 12.],
#         [12., 12.]])
# tensor([[13., 13.],
#         [13., 13.]])
# tensor([[14., 14.],
#         [14., 14.]])
# tensor([[5., 5.],
#         [5., 5.]])
# tensor([[6., 6.],
#         [6., 6.]])
# tensor([[7., 7.],
#         [7., 7.]])
# tensor([[8., 8.],
#         [8., 8.]])
# tensor([[9., 9.],
#         [9., 9.]])
# tensor([[10., 10.],
#         [10., 10.]])

### Where is this True coming from??? That is rather odd
### Why are the frames not all ones like we expect them to be?
