import numpy as np
from communication_graph import CommunicationGraph

class MarlAgent:
    def __init__(self, env, communication_graph: CommunicationGraph):
        self.env = env
        self.communication_graph = communication_graph

    