import numpy as np
from abc import ABCMeta, abstractmethod
import types


class MetaFeature:
    __metaclass__ = ABCMeta

    @abstractmethod
    def getMeta(self, in_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ColumnMeta(MetaFeature):

    def __init__(self, func, average: bool = True):
        """
        Take aggregating function, which will be applied to each column
        :param func: f(np.ndarray)->np.ndarray
        :param average: if true will output (1,1) array else (n,1) array
        """
        self.average = average
        self.function = func

    def getMeta(self, in_data: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.array([self.function(x) for x in in_data])
        if self.average:
            return np.array([np.average(result)])
        else:
            return result


if __name__ == '__main__':
    test = np.array([[2, 2, 3], [10, 10, 10]])


    def average(in_data: np.ndarray) -> np.ndarray:
        return np.average(in_data)


    metaf = ColumnMeta(average, True)
    print(metaf.getMeta(test))
