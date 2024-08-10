import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarlMNIST(gym.Env):
    metadata = {"num_agents": 8, "agent_window": 3}

    def _get_random_mnist(self):
        # TODO: have return a spaces.Box object initialized with MNIST pixel values and a ground truth value
        raise NotImplemented

    def _get_obs(self):
        # TODO: have return current image reconstruction, agent locations, and (maybe) ground truth
        raise NotImplemented

    def __init__(self, num_agents=8, agent_obs_size=4) -> None:
        MNIST_IMAGE_SIZE = 28

        self.num_agents = num_agents
        self.base_image, self.ground_truth = self._get_random_mnist()

        # Observations are dictionaries with the aggregate agent image and the target number
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "observed_image": spaces.Box(
                    0, MNIST_IMAGE_SIZE - 1, shape=(agent_obs_size,), dtype=int
                ),
                "target": spaces.Box(
                    0, MNIST_IMAGE_SIZE - 1, shape=(agent_obs_size,), dtype=int
                ),
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
        raise NotImplemented

    def step(self, actions):
        assert len(actions) == self.num_agents
        raise NotImplemented
