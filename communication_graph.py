from collections import defaultdict

from marl_agent import MarlAgent

class CommunicationGraph:
    def __init__(self, communiction_graph={}):
        self.graph = defaultdict(list)


    def add_agent(self, agent: MarlAgent, neighbors: list[MarlAgent]):
        self.graph[agent] += neighbors
        for neighbor in neighbors:
            self.graph[neighbor].append(agent)

    def remove_connection(self, agent: MarlAgent, neighbor: MarlAgent):
        self.graph[agent].remove(neighbor)
        self.graph[neighbor].remove(agent)
