import numpy as np
import torch

from meta_gan.feature_extraction.InformationMetaImpl import InformationMeta
from meta_gan.feature_extraction.StatisticalMetaImpl import StatisticalMeta


class MetaFeaturesCollector:

    def __init__(self, size: int, ):
        self.cache = {}
        self.size = size
        self.meta_features = [
            StatisticalMeta(),
            InformationMeta(size)
        ]

    def getFromCache(self, name_in: str) -> np.ndarray:
        return self.cache[name_in]

    def getLength(self):
        length = 0
        for meta in self.meta_features:
            length += meta.getLength()

        return length

    def get(self, data_in: np.ndarray, name_in: str) -> (torch.Tensor, torch.Tensor):
        (name, l_str, _) = name_in.split('_')
        if name in self.cache:
            metas, labels_length = self.cache[name]
        else:
            labels_length = int(l_str)

            meta_features = self.meta_features[0].getMeta(data_in, labels_length)
            for meta in self.meta_features[1:]:
                meta_features = np.concatenate((meta_features, meta.getMeta(data_in, labels_length)))
            self.cache[name] = (meta_features, labels_length)
            metas = meta_features
        return torch.from_numpy(metas).float(), torch.from_numpy(np.array([labels_length]))

    def getShort(self, data_in: np.ndarray, labels_length: int) -> torch.Tensor:
        meta_features = self.meta_features[0].getMeta(data_in, labels_length)
        for meta in self.meta_features[1:]:
            meta_features = np.concatenate((meta_features, meta.getMeta(data_in, labels_length)))
        metas = meta_features
        return torch.from_numpy(metas).float()


if __name__ == '__main__':
    meta = MetaFeaturesCollector(4)
    arr = np.array([[1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.1, 0.0, 0.1],
                    [1.0, 0.0, 0.1, 0.0]])
    print(meta.get(arr, "100_1_46"))
