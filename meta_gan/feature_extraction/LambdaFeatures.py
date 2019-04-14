import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class LambdaFeatures:
    models_default = [
        OneVsRestClassifier(LinearSVC(random_state=0)),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=0)
    ]

    def __init__(self, models=None, binary: bool = True, add_rf=False):
        if models is None:
            models = self.models_default
        self.cache = {}
        self.models = models
        self.binary = binary
        self.add_rf = add_rf

    def getLength(self):
        return len(self.models)

    def get(self, data_in: np.ndarray, labels_num_in: int = None, name_in: str = None,
            real: bool = True) -> torch.Tensor:
        if labels_num_in is None:
            (name, labels_num_str, _) = name_in.split('_')
            labels_num = int(labels_num_str)
        else:
            labels_num = labels_num_in

        if (name_in is not None) and (name_in in self.cache):
            lambdas = self.cache[name_in]
        else:
            x = data_in[:, labels_num:]
            if labels_num == 1:
                y = np.clip(np.around(data_in[:, :labels_num].ravel()), 0.0, 1.0)
            else:
                y = np.clip(np.around(data_in[:, :labels_num]), 0.0, 1.0)
            lambda_features = []
            for model in self.models:
                scores = cross_val_score(model, x, y, cv=KFold(3, shuffle=True, random_state=0))
                score = np.average(scores)
                lambda_features.append(score)
            av_scores = np.array(lambda_features)
            if self.binary:
                lambdas = np.zeros((self.getLength(),), dtype=float)
                lambdas[av_scores.argmax()] = 1.0
            else:
                lambdas = av_scores
            if self.add_rf:
                if real:
                    rf = np.array([0.0])
                else:
                    rf = np.array([1.0])
                lambdas = np.append(rf, lambdas)
            if name_in is not None:
                self.cache[name_in] = lambdas
        return torch.from_numpy(lambdas).float()


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    lambdas_lul = LambdaFeatures()
    results = lambdas_lul.get(np.concatenate((iris.target.reshape(-1, 1), iris.data), axis=1), 1, "lul")
    print(results)
    results = lambdas_lul.get(np.concatenate((iris.target.reshape(-1, 1), iris.data), axis=1), 1, "lul2")
    print(results)
