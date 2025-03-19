import numpy as np
from communication_graph import CommunicationGraph
from message_aggregator_interface import MessageAggregatorInterface

class MarlAgent:
    def __init__(self, message_aggregator: MessageAggregatorInterface):
        self.message_aggregator = message_aggregator

    