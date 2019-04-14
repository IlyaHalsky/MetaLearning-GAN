import numpy as np
import torch
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier


class MetaFeatures:
    def __init__(self, size: int, average_by_col=True, skewness_by_col=True, f_importanses_by_col=True):
        self.cache = {}
        self.size = size
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
            y = np.clip(np.around(data_in[:, :labels_num].ravel()), 0.0, 1.0)
        else:
            y = np.clip(np.around(data_in[:, :labels_num]), 0.0, 1.0)
        forest.fit(x, y)
        importances = forest.feature_importances_
        if self.full_importances:
            result = [0.0] * labels_num
            result.extend(importances)
            return np.array(result)
        else:
            return np.array([np.average(importances)])

    def getLength(self):
        # class labels
        length = 4
        # average labels
        if self.full_average:
            length += self.size
        else:
            length += 1
        # class/num labels
        length += self.size
        # skewness labels
        if self.full_skewness:
            length += self.size
        else:
            length += 1
        # sparsity label
        length += 1
        # importances labels
        if self.full_importances:
            length += self.size
        else:
            length += 1

        return length

    def get(self, data_in: np.ndarray, name_in: str) -> (torch.Tensor, torch.Tensor):
        (name, l_str, _) = name_in.split('_')
        if name in self.cache:
            metas, labels_length = self.cache[name]
        else:
            labels_length = int(l_str)

            meta_features = self.toBinary(labels_length)
            meta_features = np.concatenate((meta_features, self.average(data_in)))
            meta_features = np.concatenate((meta_features, self.class_or_num(data_in)))
            meta_features = np.concatenate((meta_features, self.skewness(data_in)))
            meta_features = np.concatenate((meta_features, self.sparsity(data_in)))
            meta_features = np.concatenate((meta_features, self.featureImportances(data_in, labels_length)))
            self.cache[name] = (meta_features, labels_length)
            metas = meta_features
        return torch.from_numpy(metas).float(), torch.from_numpy(np.array([labels_length]))

    def getShort(self, data_in: np.ndarray, labels_length: int) -> torch.Tensor:
        meta_features = self.toBinary(labels_length)
        meta_features = np.concatenate((meta_features, self.average(data_in)))
        meta_features = np.concatenate((meta_features, self.class_or_num(data_in)))
        meta_features = np.concatenate((meta_features, self.skewness(data_in)))
        meta_features = np.concatenate((meta_features, self.sparsity(data_in)))
        meta_features = np.concatenate((meta_features, self.featureImportances(data_in, labels_length)))
        metas = meta_features
        return torch.from_numpy(metas).float()


if __name__ == '__main__':
    meta = MetaFeatures()
    arr = np.array([[1, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    print(meta.get(arr, "100_1_46"))
