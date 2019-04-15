from abc import ABC, abstractmethod
import numpy as np


class MetaFeature(ABC):

    @abstractmethod
    def getLength(self) -> int:
        pass

    @abstractmethod
    def getMeta(self, data_in: np.ndarray, labels_size: int) -> np.ndarray:
        pass

    @staticmethod
    def isClass(data_in: np.ndarray) -> bool:
        unique, counts = np.unique(data_in, return_counts=True)
        count_dict = dict(zip(unique, counts))
        if unique.size > 2:
            return False
        if unique.size == 2 and count_dict.get(1.0, 0) > 0 and count_dict.get(0.0, 0) > 0:
            return True
        return False

    @staticmethod
    def nonClassIndexes(data_in: np.ndarray, labels_length: int) -> [int]:
        return range(labels_length, data_in.shape[1])

    @staticmethod
    def classIndexes(data_in: np.ndarray, labels_length: int) -> [int]:
        return range(0, labels_length)
