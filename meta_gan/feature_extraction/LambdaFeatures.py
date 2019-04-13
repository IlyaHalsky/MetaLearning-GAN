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

    def __init__(self, models=None):
        if models is None:
            models = self.models_default
        self.cache = {}
        self.models = models

    def getLength(self):
        return len(self.models)

    def get(self, data_in: np.ndarray, labels_num_in: int = None, name_in: str = None) -> torch.Tensor:
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
                y = data_in[:, :labels_num].ravel()
            else:
                y = data_in[:, :labels_num]
            lambda_features = []
            for model in self.models:
                scores = cross_val_score(model, x, y, cv=KFold(3, shuffle=True, random_state=0))
                score = np.average(scores)
                lambda_features.append(score)
            lambdas = np.array(lambda_features)
        return torch.from_numpy(lambdas).float()


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    lambdas_lul = LambdaFeatures()
    results = lambdas_lul.get(np.concatenate((iris.target.reshape(-1, 1), iris.data), axis=1), 1, "lul")
    print(results)
    results = lambdas_lul.get(np.concatenate((iris.target.reshape(-1, 1), iris.data), axis=1), 1, "lul2")
    print(results)
