from torchvision.datasets import MNIST
from torch.utils.data import RandomSampler
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarlMNIST(gym.Env):
    metadata = {"num_agents": 8, "agent_window": 3}

    def _get_random_mnist(self):
        # TODO: have return a spaces.Box object initialized with MNIST pixel values and a ground truth value
        random_sample, ground_truth = RandomSampler(data_source=MNIST, num_samples=1)

        return random_sample, ground_truth

    def _get_info(self):
        # TODO: return info about the state (potentially the percentage of the image that has been reconstructed)
        raise NotImplemented

    def _get_obs(self):
        # TODO: have return current image reconstruction, agent locations, and (maybe) ground truth
        raise NotImplemented

    def _get_agent_view(self, idx):
        # TODO: Use base image, agent location, agent window size to get the agent view

        agent_location = self._agent_locations[idx]
        raise NotImplemented

    def _update_composite_view(self):
        for i, agent_loc in enumerate(self._agent_locations):
            agent_view = self._get_agent_view(i)
        # TODO: compute the new composite image using agent locations. Update in place in self
        raise NotImplemented

    def __init__(self, num_agents=8, agent_obs_size=4) -> None:
        MNIST_IMAGE_SIZE = 28

        self.num_agents = num_agents
        self.agent_obs_size = agent_obs_size
        self.base_image, self.ground_truth = self._get_random_mnist()
        self._agent_locations = []
        self.vision_model = None  # TODO: initialize to cv model

        # Observations are dictionaries with the aggregate observed image and the agent locations
        # Each location is encoded as an element of {0, ..., `size` - agent_obs_size}^2 corresponding with
        # the top left corner of the agent's observation, i.e. MultiDiscrete([size- agent_obs_size, size- agent_obs_size]).
        self.observation_space = spaces.Dict(
            {
                # TODO: check if this high should be 0. Maybe we want an all black image initially as opposed to random
                "observed_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(MNIST_IMAGE_SIZE - 1, MNIST_IMAGE_SIZE - 1),
                    dtype=int,
                ),
                "agent_locations": [
                    spaces.Box(
                        0, MNIST_IMAGE_SIZE - agent_obs_size - 1, shape=(2,), dtype=int
                    )
                    for i in range(self.num_agents)
                ],
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right" for each agent
        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def reset(self, seed=None, options=None):
        # NOTE: implication here is that reset is always called since this is where self._agaent_locations is initialized
        # it may need to be that the logic to initialize agent locations should be replicated in the __init__ function as well
        super().reset(seed=seed)

        # randomize agent locations (overlap okay?)

        # Choose the agent's location uniformly at random
        agent_locations = []

        for i in range(self.num_agents):
            # TODO: check self.size
            agent_locations.append(
                self.np_random.integers(0, self.size, size=2, dtype=int)
            )

        self._agent_locations = agent_locations

        # randomize base image
        self.base_image, self.ground_truth = self._get_random_mnist()

        # return observations and info

        raise NotImplemented

    def step(self, actions):
        assert len(actions) == self.num_agents

        # update agent locations
        for i, action in enumerate(actions):
            agent_direction = self._action_to_direction(action)

            # We use `np.clip` to make sure we don't leave the grid
            self._agent_locations[i] = np.clip(
                self._agent_location + agent_direction, 0, self.size - 1
            )

            # collect observations, update composite image
            # agent_view = self._get_agent_view(self._agent_locations[i])

        # update composite image based on agent locations
        self._update_composite_view()

        # An episode is done iff the agent has reached the target
        observation = self._get_obs()
        terminated = (
            self.vision_model.predict(observation["observed_image"])
            == self.ground_truth
        )

        reward = 1 if terminated else 0  # Binary sparse rewards

        info = self._get_info()

        return observation, reward, terminated, False, info

        # return observation, reward, terminated, truncated, info
        raise NotImplemented
