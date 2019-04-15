import warnings

import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics

from meta_gan.feature_extraction.MetaFeatureApi import MetaFeature

warnings.filterwarnings("ignore", category=RuntimeWarning)

class InformationMeta(MetaFeature):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def getLength(self) -> int:
        eq_features = 1
        maxMutInfo = 1
        meanMutInfo = 1
        entropy = 1
        noiseSignal = 1
        classEnt = 1
        result = sum([eq_features, maxMutInfo, meanMutInfo, entropy, noiseSignal, classEnt])
        return result

    @staticmethod
    def mean(data_in, fr: float, to: float) -> np.ndarray:
        return np.mean(np.clip(np.nan_to_num(data_in), fr, to)) / (to - fr) + (0.5 - (to + fr) / (2.0 * (to - fr)))

    def getMeta(self, data_in: np.ndarray, labels_size: int) -> np.ndarray:
        class_index = self.classIndexes(data_in, labels_size)
        non_class_index = self.nonClassIndexes(data_in, labels_size)

        eq_count = 0
        for i in range(self.data_size):
            for j in range(i + 1, self.data_size):
                if np.array_equal(data_in[:, i], data_in[:, j]):
                    eq_count += 1
        eq_cols = self.mean(eq_count, 0.0, self.data_size)

        max_mutual_info = 0.0
        for i in class_index:
            for j in non_class_index:
                max_mutual_info = max(max_mutual_info,
                                      metrics.normalized_mutual_info_score(data_in[:, i], data_in[:, j],
                                                                           average_method='geometric'))
        max_mutual_info = self.mean(max_mutual_info, 0.0, 1.0)

        mean_mutual_info = []
        for i in class_index:
            for j in non_class_index:
                mean_mutual_info.append(metrics.normalized_mutual_info_score(data_in[:, i], data_in[:, j],
                                                                             average_method='geometric'))
        mean_mutual_info = self.mean(mean_mutual_info, 0.0, 1.0)

        entopies = []
        for i in non_class_index:
            entopies.append(stats.entropy(data_in[:, i]))
        entropy = self.mean(entopies, 0.0, 1.0)

        s_n_ration = self.mean(self.signaltonoise(data_in[:, non_class_index]), 0.0, 1.0)

        entopies_cl = []
        for i in class_index:
            entopies_cl.append(stats.entropy(data_in[:, i]))
        cl_enptropy = self.mean(entopies_cl, 0.0, 1.0)

        return np.array([eq_cols, max_mutual_info, mean_mutual_info, entropy, s_n_ration, cl_enptropy])

    @staticmethod
    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)


if __name__ == '__main__':
    meta = InformationMeta(4)
    arr = np.array([[1, 1, 0.2, 1],
                    [0.5, 0.3, 0.3, 0],
                    [0.2, 0, 0, 1]])
    print(meta.getMeta(arr, 1))
    print(meta.getLength())
