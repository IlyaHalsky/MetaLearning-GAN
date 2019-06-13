from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from meta_gan.feature_extraction.DecisionTreeMeta import DecisionTreeMeta
from meta_gan.feature_extraction.InformationMeta import InformationMeta
from meta_gan.feature_extraction.StatisticalMeta import StatisticalMeta


class MetaZerosCollector:

    def __init__(self, features_size: int, instances_size: int):
        self.cache = {}
        self.features = features_size
        self.instances = instances_size
        self.meta_features = [
            StatisticalMeta(features_size, instances_size),
            InformationMeta(features_size, instances_size),
            DecisionTreeMeta(features_size, instances_size)
        ]
        self.min_max = MinMaxScaler()
        self.length = None

    def getFromCache(self, name_in: str) -> np.ndarray:
        return self.cache[name_in]

    def getLength(self):
        if self.length is None:
            length = 0
            for meta in self.meta_features:
                length += meta.getLength()
            self.length = length
            return length
        else:
            return self.length

    def train(self, path: str):
        return 0.0

    def get(self, stacked: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.zeros((self.getLength(), 1), dtype=float)).float()

    def getShort(self, stacked: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.zeros((self.getLength(), 1), dtype=float)).float()

    def getNumpy(self, stacked: np.ndarray) -> np.ndarray:
        return np.zeros(self.getLength(), dtype=float)


if __name__ == '__main__':
    meta = MetaZerosCollector(16, 64)
    print(meta.train(f"../../processed_data/processed_16_64_2/"))
    print(meta.min_max.data_min_)
    print(meta.min_max.data_max_)
    print(meta.min_max.data_range_)
    print(meta.min_max.scale_)
