from collections import defaultdict

from marl_agent import MarlAgent

class GraphNode:
    def __init__(self, agent, value, neighbors):
        self.agent = agent
        self.value = value
        if neighbors is None:
            self.neighbors = set()
        else:
            self.neighbors = neighbors

    def get_value(self):
        return self.value

    def get_neighbors(self):
        return self.neighbors

    def get_agent(self):
        return self.agent

    def add_neighbor(self, agent):
        self.neighbors.add(agent)
        agent.neighbors.add(self)

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
        return self.graph[agent_i]
