import numpy as np
from os import walk
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pathlib import Path
import pandas as pd
import category_encoders as ce
from numpy import unravel_index


def load(name: str) -> (np.ndarray, np.ndarray):
    name_path = f'{raw_data_str}{name}/'
    data_path = f'{name_path}{name}_data.npy'
    class_path = f'{name_path}{name}_class.npy'
    data_np = np.load(data_path)
    class_np = np.load(class_path)
    return data_np, class_np


def prepareLabels(labels_in: np.ndarray) -> np.ndarray:
    if labels_in.min() == 0 and labels_in.max() == 1:
        return labels_in
    else:
        ce_bin = ce.BinaryEncoder(cols=['y'])
        df = pd.DataFrame({'y': labels_in[:, 0]})
        result = ce_bin.fit_transform(df)
        return result.values


def makeFeatures(labels_size: int, features_in: np.ndarray) -> np.ndarray:
    (_, features_in_size) = features_in.shape
    if features_in_size * (features_in_size - 1) / 2 < target_size - labels_size:
        degree = 10
        interaction_only = False
    else:
        degree = 2
        interaction_only = True
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    new_features = poly.fit_transform(features_in)
    result = new_features[:, :target_size - labels_size]
    assert result.shape[1] == target_size - labels_size
    return result


def shrinkFeatures(labels_size: int, features_in: np.ndarray, classes_in: np.ndarray) -> np.ndarray:
    clf = ExtraTreesClassifier(n_estimators=50, random_state=239)
    if classes_in.shape[1] == 1:
        classes_in_prep = classes_in.ravel()
    else:
        classes_in_prep = classes_in
    clf.fit(features_in, classes_in_prep)
    model = SelectFromModel(estimator=clf, threshold=-np.inf, prefit=True, max_features=target_size - labels_size)
    new_data = model.transform(features_in)
    if new_data.shape[1] == target_size - labels_size:
        return new_data
    else:
        new_data = features_in[:, np.random.choice(features_in.shape[1], target_size - labels_size, replace=False)]
        assert new_data.shape[1] == target_size - labels_size
        return new_data


def swapCol(data_in: np.array, labels_in: np.ndarray, frm: int, to: int) -> (np.ndarray, np.ndarray):
    data_in[:, [frm, to]] = data_in[:, [to, frm]]
    return data_in, labels_in


def swapRow(data_in: np.array, labels_in: np.ndarray, frm: int, to: int) -> (np.ndarray, np.ndarray):
    data_in[[frm, to], :] = data_in[[to, frm], :]
    labels_in[[frm, to], :] = labels_in[[to, frm], :]
    return data_in, labels_in


def findMax(data_in: np.ndarray) -> (int, int):
    return np.unravel_index(data_in.argmax(), data_in.shape)


def sortData(data_in: np.ndarray, labels_in: np.ndarray) -> (np.ndarray, np.ndarray):
    (rows_num, cols_num) = data_in.shape
    data_out = np.copy(data_in)
    labels_out = np.copy(labels_in)
    for i in range(cols_num):
        j = i
        x_max, y_max = findMax(data_out[i:, j:])
        x_max += i
        y_max += j
        (data_out, labels_out) = swapRow(data_out, labels_out, i, x_max)
        (data_out, labels_out) = swapCol(data_out, labels_out, j, y_max)

    return data_out, labels_out


def prepareData(data_in: np.ndarray, labels_in: np.ndarray) -> (np.ndarray, np.ndarray):
    data_in = data_in.astype(np.float64)
    labels_in = labels_in.astype(np.float64)
    min_max_scaler = MinMaxScaler()
    data_out = min_max_scaler.fit_transform(data_in)
    min_max_scaler = MinMaxScaler()
    labels_out = min_max_scaler.fit_transform(labels_in)
    return data_out, labels_out


def writeData(name_in: str, data_in: np.ndarray, labels_in: np.ndarray) -> bool:
    data_prepped, labels_prepped = prepareData(data_in, labels_in)
    data_sorted, labels_sorted = sortData(data_prepped, labels_prepped)
    raw_path = f'{done_data_str_raw}/{name_in}/'
    raw_name = f'{raw_path}{name_in}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    (_, labels_width) = labels_sorted.shape
    (_, data_width) = data_sorted.shape
    raw_data_name = f'{raw_name}_{data_width}_data'
    raw_labels_name = f'{raw_name}_{labels_width}_labels'
    np.save(raw_data_name, data_sorted)
    np.save(raw_labels_name, labels_sorted)
    data_out = np.concatenate((labels_sorted, data_sorted), axis=1)
    data_name = f'{done_data_str}{name_in}_{labels_width}_{data_width}'
    np.save(data_name, data_out)
    return True


def prepare(dataset_name: str, dataset_in: (np.ndarray, np.ndarray), pbar: tqdm):
    data_in, classes_in = dataset_in
    labels_in = prepareLabels(classes_in)
    (_, labels_size) = labels_in.shape
    (instances_size, features_size) = data_in.shape
    pbar.set_description("Processing %s:[%d, %d, %d]" % (dirname, instances_size, features_size, labels_size))

    if instances_size < target_size:
        choices = np.random.choice(instances_size, target_size, replace=True)
        data_in = data_in[choices]
        labels_in = labels_in[choices]
        instances_size = target_size

    labels_out = labels_in
    if features_size + labels_size < target_size:
        choices = np.random.choice(instances_size, target_size, replace=False)
        data_out = data_in[choices]
        labels_out = labels_in[choices]
        instances_size = target_size
        try:
            data_out = makeFeatures(labels_size, data_out)
        except:
            return False
    elif features_size + labels_size == target_size:
        data_out = data_in
    else:
        data_out = shrinkFeatures(labels_size, data_in, labels_in)

    choices = np.random.choice(instances_size, target_size, replace=False)
    data_to_write = data_out[choices]
    labels_to_write = labels_out[choices]
    result = writeData(dataset_name, data_to_write, labels_to_write)
    return result


if __name__ == '__main__':
    target_size = 50
    raw_data_str = "./datasets/np_raw/"
    done_data_str = f"./datasets/processed_{target_size}/"
    done_data_str_raw = f"./datasets/processed_{target_size}/raw"
    done_data_str_path = Path(f'{done_data_str}')
    done_data_str_path.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path = Path(f'{done_data_str_raw}')
    done_data_str_raw_path.mkdir(parents=True, exist_ok=True)

    start_from = 0

    non_processed = []
    for (dirpath, dirnames, filenames) in walk(raw_data_str):
        pbar = tqdm(enumerate(dirnames), total=len(dirnames))
        for i, dirname in pbar:
            if i >= start_from:
                prepare_status = prepare(dirname, load(dirname), pbar)
                if not prepare_status:
                    non_processed.append(dirname)

    print(non_processed)
