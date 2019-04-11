import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifierCV


class LambdaFeatures:
    models = [
        RidgeClassifierCV(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=0)
    ]

    def __init__(self):
        self.cache = {}

    def get(self, data_in: np.ndarray, labels_num_in: int = None, name_in: str = None) -> np.ndarray:
        if labels_num_in is None:
            (name, labels_num_str, _) = name_in.split('_')
            labels_num = int(labels_num_str)
        else:
            labels_num = labels_num_in

        if (name_in is not None) and (name_in in self.cache):
            return self.cache[name_in]
        else:
            x = data_in[:, labels_num:]
            if labels_num == 1:
                y = data_in[:, :labels_num].ravel()
            else:
                y = data_in[:, :labels_num]
            lambda_features = []
            for model in self.models:
                scores = cross_val_score(model, x, y, cv=3)
                score = np.average(scores)
                lambda_features.append(score)
            return np.array(lambda_features)


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    lambdas = LambdaFeatures()
    results = lambdas.get(np.concatenate((iris.target.reshape(-1, 1), iris.data), axis=1), 1, "lul")
    print(results)
    results = lambdas.get(np.concatenate((iris.target.reshape(-1, 1), iris.data), axis=1), 1, "lul2")
    print(results)
