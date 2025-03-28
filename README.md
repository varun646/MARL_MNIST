# Multi-Agent RL for MNIST Image Identification

This repository implements a multi-agent reinforcement learning (RL) system to identify MNIST images using the custom Gymnasium environment [Multi-Agent MNIST Reconstruction](https://github.com/yourusername/multi-agent-mnist-gym). The RL agents work together to reconstruct and identify an MNIST image in as few steps as possible. A computer vision attention model is used to evaluate the reconstructed image and guide the training process.

## Overview

### Project Description

The project aims to train multiple RL agents to collaboratively identify an MNIST image. The agents are placed on random parts of the image and have limited views. They navigate the image using actions (up, down, left, right), and their observations are combined to form a reconstructed image. A computer vision attention model is then used to evaluate how well the reconstructed image matches the original MNIST image. The RL system is trained to minimize the number of steps needed to accurately identify the image.

### Features

- **Multi-Agent RL:** Each agent learns to navigate the image optimally, contributing to the overall image reconstruction.
- **Attention Model:** A computer vision attention model evaluates the reconstructed image, providing feedback for the RL agents.
- **Efficient Training:** The system is designed to minimize the number of steps needed for accurate image identification.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/) (for the attention model)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (for RL algorithms)
- NumPy
- OpenCV (optional, for visualization)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/varun646/MARL_MNIST.git
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the custom environment:

    ```bash
    pip install -e .
    ```

## Enhancements over Existing Literature

### Implementation of Attention-Based Filtering Module
Add notion of attention-based filtering to decentralized communication network. This way, instead of treating all messages as equally important, we can focus more on input that provides new insights

### Implementation of Graph Attention Network for Communication Graph
Replace a fully-connected communication graph with a graph attention network so we can learn to devalue less important/less reliable agents.

**Future Testing of Efficacy:** implement an adversarial/unreliable agent (one that either actively works against the system or provides output at random) and test how the system compares to a system without GAT (NOTE: from a SWE standpoint, we will need to make the communication methods modular - decentralized equally weighted, attention-based filtering, or GAT)


## TODO:
- add infra for custom communication graph and have agents share info in distributed manner between steps