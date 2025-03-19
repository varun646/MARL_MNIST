from message_aggregator_interface import MessageAggregatorInterface
import numpy as np

class MessageAverager(MessageAggregatorInterface):
    def __init__(self):
        pass

    @staticmethod
    def aggregate_messages(messages):
        return np.mean(messages, axis=0)
