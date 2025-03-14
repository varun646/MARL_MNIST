from collections import defaultdict

from marl_agent import MarlAgent

class GraphNode:
    def __init__(self, agent: MarlAgent, value):
        self.agent = agent
        self.value = value

class CommunicationGraph:
    def __init__(self, communication_graph=None):
        if communication_graph is None:
            self.communication_graph = {}
        self.graph = defaultdict(list)


    def add_agent(self, agent: MarlAgent, neighbors: list[MarlAgent]):
        self.graph[agent] += neighbors
        for neighbor in neighbors:
            self.graph[neighbor].append(agent)

    def remove_connection(self, agent: MarlAgent, neighbor: MarlAgent):
        self.graph[agent].remove(neighbor)
        self.graph[neighbor].remove(agent)

    def get_message(self, agent_i: GraphNode):
        return agent_i.value
