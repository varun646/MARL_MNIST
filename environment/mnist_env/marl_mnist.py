from random import random

from numpy.ma.core import shape
import torchvision.datasets as datasets
from numpy.random import random_sample
from torch.utils.data import RandomSampler, DataLoader
from torchvision.transforms import ToTensor
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarlMNIST(gym.Env):
    metadata = {"num_agents": 8, "agent_window": 3}

    def _get_random_mnist(self):
        # TODO: check why transform still returns tensor
        mnist = datasets.MNIST(
            root="data", download=True, transform=lambda x: np.array(x)
        )
        sampler = RandomSampler(data_source=mnist, num_samples=1, replacement=True)
        dataloader = DataLoader(dataset=mnist, sampler=sampler, batch_size=1)
        random_img, ground_truth = next(iter(dataloader))
        ground_truth = ground_truth.item()

        return random_img, ground_truth

    def _get_info(self):
        """
        Return percentage of base image that has been reconstructed

        :return: float : percentage uncovered
        """
        num_non_zero = np.count_nonzero(self.observed_image)

        return num_non_zero / (self.env_size * self.env_size)

    def _get_info(self):
        # TODO: return info about the state (potentially the percentage of the image that has been reconstructed)
        raise NotImplemented

    def _get_obs(self):
        return self.observed_image, self._agent_locations

    def _get_agent_view(self, idx):
        """
        Get numpy array representing only view of agent

        :param idx: agent index
        :return: np.ndarray of shape (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), with agent view populated
        """

        agent_location = self._agent_locations[idx]
        agent_x = agent_location[0]
        agent_y = agent_location[1]

        partial_agent_view = self.base_image[
            agent_x : agent_x + self.agent_obs_size,
            agent_y : agent_y + self.agent_obs_size,
        ]

        agent_view = np.zeros(shape=shape(self.base_image), dtype=np.uint8)
        agent_view[agent_y : agent_y + self.agent_obs_size] = partial_agent_view

        return agent_view

    def _update_composite_view(self):
        new_composite = self.observed_image  # current composite
        for i, agent_loc in enumerate(self._agent_locations):
            agent_view = self._get_agent_view(i)  # TODO: implement this
            new_composite = np.where(new_composite != 0, new_composite, agent_view)

        self.observed_image = new_composite

    def __init__(self, num_agents=8, agent_obs_size=4) -> None:
        MNIST_IMAGE_SIZE: int = 28

        self.env_size: int = MNIST_IMAGE_SIZE
        self.num_agents: int = num_agents
        self.agent_obs_size: int = agent_obs_size
        self.base_image, self.ground_truth = self._get_random_mnist()
        self._agent_locations = []
        self.vision_model = None  # TODO: initialize to cv model
        self.observed_image = np.zeros(shape=shape(self.base_image), dtype=np.uint8)

        # Observations are dictionaries with the aggregate observed image and the agent locations
        # Each location is encoded as an element of {0, ..., `size` - agent_obs_size}^2 corresponding with
        # the top left corner of the agent's observation, i.e. MultiDiscrete([size- agent_obs_size, size- agent_obs_size]).
        agent_locations_dict = {}

        for i in range(self.num_agents):
            agent_locations_dict[str(i)] = spaces.Box(
                0, MNIST_IMAGE_SIZE - agent_obs_size - 1, shape=(2,), dtype=np.uint8
            )

        self.observation_space = spaces.Dict(
            {
                "observed_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE),
                    dtype=np.uint8,
                ),
                "agent_locations": spaces.Dict(agent_locations_dict),
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
        # it may be clearer for the logic to initialize agent locations should be replicated in the __init__ function as well
        # although not necessary since according to the gymnasium documentation, reset() is always called before step()
        super().reset(seed=seed)

        # randomize agent locations (overlap okay?)

        # Choose the agent's location uniformly at random
        agent_locations = []

        for i in range(self.num_agents):
            agent_locations.append(
                self.np_random.integers(0, self.env_size, size=2, dtype=np.uint8)
            )

        self._agent_locations = agent_locations

        # randomize base image
        self.base_image, self.ground_truth = self._get_random_mnist()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, actions):
        assert len(actions) == self.num_agents

        # update agent locations
        for i, action in enumerate(actions):
            agent_direction = self._action_to_direction[action]

            # We use `np.clip` to make sure we don't leave the grid
            self._agent_locations[i] = np.clip(
                self._agent_locations[i] + agent_direction, 0, self.env_size - 1
            )

        # update composite image based on agent locations
        self._update_composite_view()

        # An episode is done iff the cv model has correctly predicted the base image
        observation = self._get_obs()
        terminated = self.vision_model.predict(self.observed_image) == self.ground_truth

        reward = 1 if terminated else 0  # Binary sparse rewards

        info = self._get_info()

        # return observation, reward, terminated, truncated, info
        return observation, reward, terminated, False, info

    def render(self):
        # TODO: display base image, dots at agent locations

        raise NotImplemented

    def render(self):
        # TODO: display base image, dots at agent locations

        raise NotImplemented
