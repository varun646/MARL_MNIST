from environment.mnist_env.marl_mnist import MarlMNIST

env = MarlMNIST()

for _ in range(10):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
