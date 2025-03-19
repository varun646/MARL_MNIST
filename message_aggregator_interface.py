from abc import ABC, abstractmethod

class MessageAggregatorInterface(ABC):
    @abstractmethod
    @staticmethod
    def aggregate_messages(messages):
        pass