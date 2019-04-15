import numpy as np
import scipy.stats as stats

from meta_gan.feature_extraction.MetaFeatureApi import MetaFeature


class StatisticalMeta(MetaFeature):
    def getLength(self) -> int:
        variation = 1
        kurtosis = 1
        pearson = 1
        skew = 1
        std = 1
        result = sum([variation, kurtosis, pearson, skew, std])
        return result

    @staticmethod
    def mean(data_in, fr: float, to: float) -> np.ndarray:
        return np.mean(np.clip(np.nan_to_num(data_in), fr, to)) / (to - fr) + (0.5 - (to + fr) / 2.0)

    def getMeta(self, data_in: np.ndarray, labels_size: int) -> np.ndarray:
        non_class_indexes = self.nonClassIndexes(data_in, labels_size)

        variation = self.mean(stats.variation(data_in[:, non_class_indexes], 0), 0.0, 1.0)

        kurtosis = self.mean(stats.kurtosis(data_in[:, non_class_indexes], 0), -3.0, 3.0)

        pearsons = []
        for i, index in enumerate(non_class_indexes):
            for j in range(i + 1, len(non_class_indexes)):
                pearsons.append(stats.pearsonr(data_in[:, i], data_in[:, j])[0])
        pearson = self.mean(pearsons, -1.0, 1.0)

        skew = self.mean(stats.skew(data_in[:, non_class_indexes], axis=0), -2.0, 2.0)

        std = self.mean(np.std(data_in[:, non_class_indexes], axis=0), 0.0, 1.0)

        return np.array([variation, kurtosis, pearson, skew, std])


if __name__ == '__main__':
    meta = StatisticalMeta()
    arr = np.array([[1, 1, 1, 1],
                    [0.2, 0.3, 0, 0],
                    [0.2, 0, 0, 1]])
    print(meta.getMeta(arr, 1))
    print(meta.getLength())
