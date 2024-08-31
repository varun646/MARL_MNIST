# Custom Gymnasium Environment: Multi-Agent MNIST Gymnasium Environment

This repository contains a custom environment for the [Gymnasium](https://gymnasium.farama.org/) library, inspired by OpenAI's Gym. The environment simulates multiple agents navigating an MNIST image and collaboratively reconstructing it based on their observations. The goal is to predict the value of the original MNIST image using the minimum number of observations of the agents, and rewards are provided based on the accuracy of the reconstruction.

## Overview

### Environment Description

- **Base Image:** The environment starts with a random MNIST image selected as the base image.
- **Agents:** There are multiple agents (default is 8) each initialized at random locations on the MNIST image. Each agent has a limited 4x4 view of the image.
- **Actions:** Each agent can take one of four actions:
  - `0`: Move up
  - `1`: Move down
  - `2`: Move left
  - `3`: Move right
- **Rewards:** The agents' goal is to reconstruct the MNIST image from their combined observations. A Convolutional Neural Network (CNN) predicts the reconstructed image, and the reward is calculated based on how closely this prediction matches the original MNIST image.

### Features

- **Multi-Agent System:** Allows for cooperative behavior as agents work together to reconstruct the image.
- **Sparse Rewards:** Rewards are only given when the CNN's prediction matches the ground truth, promoting efficient and accurate navigation by the agents.
- **Customizable:** The number of agents, the size of the view, and the action space can be easily adjusted to fit different experimental setups.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Gymnasium](https://gymnasium.farama.org/)
- TensorFlow or PyTorch (for the CNN model)
- NumPy
- OpenCV (optional, for visualization)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/varun646/MARL_MNIST.git
    cd environment
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the custom environment:

    ```bash
    pip install -e .
    ```

## Usage

### Creating the Environment

You can create the environment using Gymnasium's `make` function:

```python
import gymnasium as gym
import marl_mnist  # Make sure this module is properly imported

env = gym.make('MultiAgentMNIST-v0')
```
