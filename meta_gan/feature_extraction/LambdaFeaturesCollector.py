from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class LambdaFeaturesCollector:
    models_default = [
        SVC(kernel="linear", C=0.025, random_state=0),
        KNeighborsClassifier(n_neighbors=3),
        SVC(gamma=2, C=1, random_state=0),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        GaussianNB()
    ]

    def __init__(self, features_size: int, instances_size: int, models=None, binary: bool = True):
        if models is None:
            models = self.models_default
        self.cache = {}
        self.models = models
        self.binary = binary
        self.features = features_size
        self.instances = instances_size
        self.jobs = None

    def getLength(self):
        return len(self.models)

    @staticmethod
    def data(zero_in: np.ndarray, one_in: np.ndarray) -> np.ndarray:
        return np.append(zero_in, one_in, axis=0)

    def labels(self) -> np.ndarray:
        return np.append(np.zeros(self.instances), np.ones(self.instances))

    def get(self, stacked: np.ndarray, name_in: str = None) -> torch.Tensor:
        if (name_in is not None) and (name_in in self.cache):
            lambdas = self.cache[name_in]
        else:
            x = self.data(stacked[0], stacked[1])
            y = self.labels()
            lambda_features = []
            for model in self.models:
                scores = cross_val_score(model, x, y, cv=KFold(3, shuffle=True, random_state=0), n_jobs=self.jobs)
                score = np.average(scores)
                lambda_features.append(score)
            av_scores = np.array(lambda_features)
            if self.binary:
                min_max = MinMaxScaler()
                lambdas = min_max.fit_transform(av_scores.reshape(-1, 1))
            else:
                lambdas = av_scores
            if name_in is not None:
                self.cache[name_in] = lambdas
        return torch.from_numpy(lambdas).float()


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    lambdas_lul = LambdaFeaturesCollector(16, 64)
    lambdas_lul.jobs = 3
    path = "../../processed_data/processed_16_64_2/"
    only_files = [f for f in listdir(path) if isfile(join(path, f))]
    results = []
    for name in tqdm(only_files):
        stacked = np.load(f'{path}{name}')
        results.append(lambdas_lul.get(stacked, name).numpy())
    results = np.array(results)
    print(results)
    print(np.mean(results, axis=0))
