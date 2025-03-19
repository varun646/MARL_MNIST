import logging

from agent import RandomPolicy
from marl_agent import MarlAgent
from message_averager import MessageAverager
from utils import config


from environment.mnist_env.marl_mnist import MarlMNIST

# Logging
now = config.Now()
log = logging.getLogger('root')
log.setLevel('INFO')
log.addHandler(config.MyHandler())

class MultiAgentSystemWrapper:
    def __init__(self, env, communication_graph, agents):
        self.env = env
        self.communication_graph = communication_graph
        self.agents = agents


if __name__ == '__main__':
    num_agents = 4
    horizon = 10

    env = MarlMNIST(num_agents=num_agents)
    random_policy = RandomPolicy(action_dim=env.action_space.shape[0], num_agents=env.num_agents)
    message_aggregator = MessageAverager()

    agent1 = MarlAgent(message_aggregator=message_aggregator)
    agent2 = MarlAgent(message_aggregator=message_aggregator)
    agent3 = MarlAgent(message_aggregator=message_aggregator)
    agent4 = MarlAgent(message_aggregator=message_aggregator)




