import numpy as np
import torch
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler


class MetaFeatures:
    def __init__(self, average_by_col=True, skewness_by_col=True, f_importanses_by_col=True):
        self.cache = {}
        self.full_average = average_by_col
        self.full_skewness = skewness_by_col
        self.full_importances = f_importanses_by_col

    def getFromCache(self, name_in: str) -> np.ndarray:
        return self.cache[name_in]

    def toBinary(self, number: int) -> np.ndarray:
        return np.array([(number >> k) & 1 for k in range(0, 4)])

    def average(self, data_in: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.average(data_in, axis=0)
        if not self.full_average:
            return np.average(result)
        else:
            return result

    def class_or_num(self, data_in: np.ndarray) -> np.ndarray:
        result = []
        for col in data_in.T:
            unique = np.unique(col)
            if unique.size > 2:
                result.append(0)
            else:
                result.append(1)
        return np.array(result)

    def skewness(self, data_in: np.ndarray) -> np.ndarray:
        result_in = np.nan_to_num(skew(data_in, axis=0))
        min_max_scaler = MinMaxScaler()
        result_out = min_max_scaler.fit_transform(result_in.reshape(-1, 1)).ravel()
        if not self.full_skewness:
            return np.average(result_out)
        else:
            return result_out

    @staticmethod
    def sparsity(data_in: np.ndarray) -> np.ndarray:
        sparsity = 1.0 - (np.count_nonzero(data_in) / float(data_in.size))
        return np.array([sparsity])

    def featureImportances(self, data_in: np.ndarray) -> np.ndarray:

    def get(self, data_in: np.ndarray, name_in: str) -> torch.Tensor:
        (name, l_str, d_str) = name_in.split('_')
        labels_length = int(l_str)

        meta_features = self.toBinary(labels_length)
        meta_features = np.concatenate((meta_features, self.average(data_in)))
        meta_features = np.concatenate((meta_features, self.class_or_num(data_in)))
        meta_features = np.concatenate((meta_features, self.skewness(data_in)))
        meta_features = np.concatenate((meta_features, self.skewness(data_in)))
        meta_features = np.concatenate((meta_features, self.sparsity(data_in)))
        return meta_features


if __name__ == '__main__':
    meta = MetaFeatures()
    arr = np.array([[1, 0, 1, 0],
                    [1, 0.5, 0, 0],
                    [1, 0.7, 0, 0]])
    print(meta.get(arr, "100_8_46"))
