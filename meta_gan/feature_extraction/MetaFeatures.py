import numpy as np
import torch
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier


class MetaFeatures:
    def __init__(self, average_by_col=True, skewness_by_col=True, f_importanses_by_col=True):
        self.cache = {}
        self.full_average = average_by_col
        self.full_skewness = skewness_by_col
        self.full_importances = f_importanses_by_col

    def getFromCache(self, name_in: str) -> np.ndarray:
        return self.cache[name_in]

    @staticmethod
    def toBinary(number: int) -> np.ndarray:
        return np.array([(number >> k) & 1 for k in range(0, 4)])

    def average(self, data_in: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.average(data_in, axis=0)
        if not self.full_average:
            return np.array([np.average(result)])
        else:
            return result

    @staticmethod
    def class_or_num(data_in: np.ndarray) -> np.ndarray:
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
            return np.array([np.average(result_out)])
        else:
            return result_out

    @staticmethod
    def sparsity(data_in: np.ndarray) -> np.ndarray:
        sparsity = 1.0 - (np.count_nonzero(data_in) / float(data_in.size))
        return np.array([sparsity])

    def featureImportances(self, data_in: np.ndarray, labels_num: int) -> np.ndarray:
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)
        x = data_in[:, labels_num:]
        if labels_num == 1:
            y = data_in[:, :labels_num].ravel()
        else:
            y = data_in[:, :labels_num]
        forest.fit(x, y)
        importances = forest.feature_importances_
        if not self.full_importances:
            return np.array([np.average(importances)])
        else:
            return importances

    def get(self, data_in: np.ndarray, name_in: str) -> torch.Tensor:
        (name, l_str, _) = name_in.split('_')
        if name in self.cache:
            return self.cache[name]
        else:
            labels_length = int(l_str)

            meta_features = self.toBinary(labels_length)
            meta_features = np.concatenate((meta_features, self.average(data_in)))
            meta_features = np.concatenate((meta_features, self.class_or_num(data_in)))
            meta_features = np.concatenate((meta_features, self.skewness(data_in)))
            meta_features = np.concatenate((meta_features, self.sparsity(data_in)))
            meta_features = np.concatenate((meta_features, self.featureImportances(data_in, labels_length)))
            self.cache[name] = meta_features
            return torch.from_numpy(meta_features).float()


if __name__ == '__main__':
    meta = MetaFeatures()
    arr = np.array([[1, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    print(meta.get(arr, "100_1_46"))
