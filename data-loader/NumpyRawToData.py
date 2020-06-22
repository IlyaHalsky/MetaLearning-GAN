import numpy as np
from os import walk
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pathlib import Path
import pandas as pd
import category_encoders as ce
from numpy import unravel_index
import math
import random
import time
from scipy.stats import spearmanr

import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")


# function that returns a random path given a total number of cities
def random_path(no_cities, seed1):
    tour = list(range(no_cities))
    random.seed(seed1)
    random.shuffle(tour)
    return tour


# Given a list of cities, calculates the cost of the tour
def tour_cost(tours, cost_fun, data_in, distances):
    total_cost = 0
    cost_i = 0
    n = len(tours)
    for i, city in enumerate(tours):
        if i == n - 1:
            continue
        else:
            if (cost_fun == "d"):
                cost_i = distance(data_in[tours[i]], data_in[tours[i + 1]])
            total_cost = total_cost + cost_i

    return total_cost


# mutation operator that swaps two cities randomly to create a new path

def mutation_operator(tours):
    r1 = list(range(len(tours)))
    r2 = list(range(len(tours)))
    random.shuffle(r1)
    random.shuffle(r2)
    for i in r1:
        for j in r2:
            if i < j:
                next_state = tours[:]
                next_state[i], next_state[j] = tours[j], tours[i]
                yield next_state


# probabilistically choosing a neighbour
# def Probability_acceptance(prev_score, next_score, temperature):
#     if next_score < prev_score:
#         return 1.0
#     elif temperature == 0:
#         return 0.0
#     else:
#         return math.exp(-abs(next_score - prev_score) / temperature)


# The cooling schedule based on  kirkpatrick model
# def cooling_schedule(start_temp, cooling_constant):
#     T = start_temp
#     while True:
#         yield T
#         T = cooling_constant * T


# This function implements randomized hill climbing for TSP
def randomized_hill_climbing(no_cities, cost_func, MEB, seed1, data_in, distances):
    dict1 = {}
    best_path = random_path(no_cities, seed1)
    best_cost = tour_cost(best_path, cost_func, data_in, distances)
    evaluations_count = 1
    while evaluations_count < MEB:
        for next_city in mutation_operator(best_path):
            if evaluations_count == MEB:
                break
            str1 = ''.join(str(e) for e in next_city)
            # Skip calculating the cost of repeated paths
            if str1 in dict1:
                evaluations_count += 1
                continue

            next_tCost = tour_cost(next_city, cost_func, data_in, distances)
            # store it in the dictionary
            dict1[str1] = next_tCost
            evaluations_count += 1

            # selecting the path with lowest cost
            if next_tCost < best_cost:
                best_path = next_city
                best_cost = next_tCost

    return best_cost, best_path, evaluations_count


# This function implements simulated annealing for TSP
# def simulated_annealing(no_cities, cost_func, MEB, seed1):
#     start_temp = 70
#     cooling_constant = 0.9995
#     best_path = None
#     best_cost = None
#     current_path = random_path(int(no_cities), seed1)
#     current_cost = tour_cost(current_path, cost_func, data_in)
#
#     if best_path is None or current_cost < best_cost:
#         best_cost = current_cost
#         best_path = current_path
#
#     num_evaluations = 1
#     temp_schedule = cooling_schedule(start_temp, cooling_constant)
#     for temperature in temp_schedule:
#         flag = False
#         # examinning moves around our current path
#         for next_path in mutation_operator(current_path):
#             if num_evaluations == MEB:
#                 # print "reached meb"
#                 flag = True
#                 break
#
#             next_cost = tour_cost(next_path, cost_func, data_in)
#
#             if best_path is None or next_cost < best_cost:
#                 best_cost = next_cost
#                 best_path = next_path
#
#             num_evaluations += 1
#             p = Probability_acceptance(current_cost, next_cost, temperature)
#             if random.random() < p:
#                 current_path = next_path
#                 current_cost = next_cost
#                 break
#
#         if flag:
#             break
#
#     return best_path, best_cost, num_evaluations


############################ TSPH ##############################
def TSP(data_in):
    data = data_in[:, :]
    distances = {}
    for i in data:
        for j in data:
            distances[(str(i), str(j))] = distance(i, j)
    no_cities = len(data)
    if no_cities > 100:
        MEB = 200000
    else:
        MEB = 20000
    dict = {}
    cost_func = "d"
    seed1 = 26
    best_cost = None
    best_path = None
    num_evaluations = None
    best_cost, best_path, num_evaluations = randomized_hill_climbing(no_cities, cost_func, MEB, seed1, data, distances)

    data_in = data_in.T
    np.append(best_path, 16)
    data_in = data_in[:, best_path]

    data = data_in[:, :]
    best_cost = None
    best_path = None
    num_evaluations = None
    best_cost, best_path, num_evaluations = randomized_hill_climbing(no_cities, cost_func, MEB, seed1, data, distances)
    data_in = data_in[best_path, :]

    out_zeros = data_in[:64][:]
    out_ones = data_in[64:][:]

    return out_zeros, out_ones


def load(name: str) -> (np.ndarray, np.ndarray):
    name_path = f'{raw_data_str}{name}/'
    data_path = f'{name_path}{name}_data.npy'
    class_path = f'{name_path}{name}_class.npy'
    data_np = np.load(data_path)
    class_np = np.load(class_path)
    return data_np, class_np


def prepareLabels(labels_in: np.ndarray) -> np.ndarray:
    onehot = OneHotEncoder(sparse=False, categories='auto')
    result = onehot.fit_transform(labels_in)
    return result


# def makeFeatures(labels_size: int, features_in: np.ndarray) -> np.ndarray:
#     (_, features_in_size) = features_in.shape
#     if features_in_size * (features_in_size - 1) / 2 < target_size - labels_size:
#         degree = 10
#         interaction_only = False
#     else:
#         degree = 2
#         interaction_only = True
#     poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
#     new_features = poly.fit_transform(features_in)
#     result = new_features[:, :target_size - labels_size]
#     assert result.shape[1] == target_size - labels_size
#     return result

def scaleData(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    data = np.append(zero_in, one_in, axis=0)
    data = data.astype(np.float64)
    min_max_scaler = MinMaxScaler()
    data_out = min_max_scaler.fit_transform(data)
    return data_out[:target_instances], data_out[target_instances:]


def shrinkFeatures(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=239)
    data = np.append(zero_in, one_in, axis=0)
    labels = np.append(np.zeros(target_instances), np.ones(target_instances))
    clf.fit(data, labels)
    model = SelectFromModel(estimator=clf, threshold=-np.inf, prefit=True, max_features=target_features)
    new_data = model.transform(data)
    if new_data.shape[1] == target_features:
        return new_data[:target_instances], new_data[target_instances:]
    else:
        given_features = data.shape[1]
        new_data = data[:, np.random.choice(given_features.shape[1], target_features, replace=False)]
        assert new_data.shape[1] == target_features
        return new_data[:target_instances], new_data[target_instances:]


def swapCol(data_in: np.array, frm: int, to: int) -> np.ndarray:
    data_in[:, [frm, to]] = data_in[:, [to, frm]]
    return data_in


def swapRow(data_in: np.array, frm: int, to: int) -> np.ndarray:
    data_in[[frm, to], :] = data_in[[to, frm], :]
    return data_in


def swapRows(data_in: np.array, frm: np.ndarray, to: np.ndarray, col_num: int) -> np.ndarray:
    to_sorted = np.argsort(data_in[to, col_num])[::-1]
    for i, j in enumerate(to_sorted):
        frm_i = frm[i]
        to_i = to[j]
        data_in[[frm_i, to_i], :] = data_in[[to_i, frm_i], :]
    return data_in


def findMax(zero_in: np.ndarray, one_in: np.ndarray, step: int) -> (int, np.ndarray, np.ndarray):
    max_col_mean = []
    (rows, cols) = zero_in.shape
    for i in range(cols):
        zero_col = zero_in[:, i]
        zero_max_ind = np.argpartition(zero_col, -step)[-step:]
        if rows != step:
            zero_av = np.mean(np.delete(zero_col, zero_max_ind))
        else:
            zero_av = 0.0
        zero_val = np.mean(zero_col[zero_max_ind]) - zero_av
        one_col = one_in[:, i]
        one_max_ind = np.argpartition(one_col, -step)[-step:]
        if rows != step:
            one_av = np.mean(np.delete(one_col, one_max_ind))
        else:
            one_av = 0.0
        one_val = np.mean(one_col[one_max_ind]) - one_av
        max_col_mean.append((one_val + zero_val) / 2)

    max_col_mean = np.array(max_col_mean)
    max_col = np.argmax(max_col_mean).item(0)
    zero_max_ind = np.argpartition(zero_in[:, max_col], -step)[-step:]
    one_max_ind = np.argpartition(one_in[:, max_col], -step)[-step:]
    return max_col, zero_max_ind, one_max_ind


############################ DIAG ##############################
def sortData(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    zero_out = np.copy(zero_in)
    one_out = np.copy(one_in)
    step = target_instances // target_features
    for col_num in range(target_features):
        row_num = col_num * step
        col_max, zero_max, one_max = findMax(zero_out[row_num:, col_num:], one_out[row_num:, col_num:], step)
        col_max += col_num
        rows_from = np.array(range(row_num, row_num + 4))
        zero_max = zero_max + row_num
        zero_out = swapCol(zero_out, col_num, col_max)
        zero_out = swapRows(zero_out, rows_from, zero_max, col_num)
        one_max = one_max + row_num
        one_out = swapCol(one_out, col_num, col_max)
        one_out = swapRows(one_out, rows_from, one_max, col_num)

    return zero_out, one_out


def distance(v1, v2):
    ans = 0
    for i, j in zip(v1, v2):
        ans += (i - j) * (i - j)
    return ans


# def get_way(qwerty, n, obj_or_feat='obj'):
#     M = np.zeros([n, n])
#     for i in np.arange(0, n, 1):
#         for j in np.arange(0, n, 1):
#             if i != j:
#                 if obj_or_feat == 'obj':
#                     M[i, j] = distance(qwerty.iloc[i], qwerty.iloc[j])
#                 else:
#                     M[i, j] = distance(qwerty.loc[:, 'A%d' % (i + 1)], qwerty.loc[:, 'A%d' % (j + 1)])
#             else:
#                 M[i, j] = float('inf')
#     way = []
#     way.append(0)
#     for i in np.arange(1, n, 1):
#         s = []
#         for j in np.arange(0, n, 1):
#             s.append(M[way[i - 1], j])
#         way.append(s.index(min(s)))      # Индексы пунктов ближайших городов соседей
#         for j in np.arange(0, i, 1):
#             M[way[i], way[j]] = float('inf')
#             M[way[i], way[j]] = float('inf')
#     #print(way)
#     if obj_or_feat == 'obj':
#         S = np.sum([distance(qwerty.iloc[way[i]], qwerty.iloc[way[i + 1]]) for i in np.arange(0, n - 1, 1)]) + distance(qwerty.iloc[n - 1], qwerty.iloc[0])
#     else:
#         S = np.sum([distance(qwerty.loc[:, 'A%d' % (way[i] + 1)], qwerty.loc[:, 'A%d' % (way[i + 1] + 1)]) for i in np.arange(0, n - 1, 1)]) + distance(qwerty.loc[:, 'A%d' % n], qwerty.loc[:, 'A1'])
#     print(S)
#     #print(data_copy)
#     return way, S


############################ TSPG ##############################

############################ TSPG ##############################
def greedy_salesman_search(data_in: np.ndarray):
    extra_row = np.zeros((1, len(data_in.T)))
    data_in = np.concatenate((extra_row, data_in), axis=0)

    extra_col = np.zeros((len(data_in), 1))
    data_in = np.concatenate((extra_col, data_in), axis=1)

    classes = data_in[:, -1]
    classes = np.reshape(classes, (len(classes), 1))

    n = len(data_in.T) - 1
    qwerty = data_in[:, :-1].T
    visited = [0]
    last_obj = qwerty[0]
    for i in range(1, n):
        cur_index = -1
        cur_cost = 1000000000
        for j in range(0, n):
            if j not in visited:
                cost = distance(last_obj, qwerty[j])
                if cost < cur_cost:
                    cur_cost = cost
                    cur_index = j
        if cur_index != -1:
            visited.append(cur_index)
            last_obj = qwerty[cur_index]
    visited.append(17)

    sorted_data = data_in[:, visited]
    sorted_data = sorted_data[:, :-1]

    sorted_data0 = sorted_data[:65, :]
    sorted_data1 = np.concatenate((extra_row, sorted_data[65:, :]), axis=0)
    n = len(sorted_data0)
    visited = [0]
    last_obj = sorted_data0[0]
    for i in range(1, n):
        cur_index = -1
        cur_cost = 1000000000
        for j in range(0, n):
            if j not in visited:
                cost = distance(last_obj, sorted_data0[j])
                if cost < cur_cost:
                    cur_cost = cost
                    cur_index = j
        if cur_index != -1:
            visited.append(cur_index)
            last_obj = sorted_data0[cur_index]
    sorted_data0 = sorted_data0[visited, :]


    n = len(sorted_data1)
    visited = [0]
    last_obj = sorted_data1[0]
    for i in range(1, n):
        cur_index = -1
        cur_cost = 1000000000
        for j in range(0, n):
            if j not in visited:
                cost = distance(last_obj, sorted_data1[j])
                if cost < cur_cost:
                    cur_cost = cost
                    cur_index = j
        if cur_index != -1:
            visited.append(cur_index)
            last_obj = sorted_data1[cur_index]

    sorted_data1 = sorted_data1[visited, :]
    sorted_data = np.concatenate((sorted_data0, sorted_data1[1:, :]), axis=0)
    sorted_data = np.concatenate((sorted_data, classes), axis=1)
    sorted_data = sorted_data[1:, 1:]

    out_zeros = sorted_data[np.where(sorted_data[:, 16] == 0.0)]
    out_ones = sorted_data[np.where(sorted_data[:, 16] == 1.0)]
    return out_zeros[:, :-1], out_ones[:, :-1]


# def salesmanMethod(data_in: np.ndarray) -> np.ndarray:
#     n = len(data_in)
#     row_indexes = np.arange(n)
#     i = 0
#     way, _ = get_way(data_in.iloc[:, :-1], n)
#     # print(way)
#     for v in way:
#         np.put(row_indexes, i, v)
#         i += 1
#     data_copy = np.copy(data_in)
#     data_copy['A18'] = row_indexes
#     sorted_data_copy = data_copy.sort_values('A18')
#     # print(sorted_data_copy)
#     row_indexes = np.arange(16)
#     i = 0
#     way, _ = get_way(data_in.iloc[:, :-1], 16, 'features')
#     # print(way)
#     sorted_indexes = []
#     for v in way:
#         np.put(row_indexes, i, v)
#         sorted_indexes.append('A%d' % (v + 1))
#         i += 1
#     sorted_indexes.append('A17')
#     sorted_data = sorted_data_copy.reindex(columns=sorted_indexes)
#     return sorted_data


def pearson(pairs):
  product_sum = 0.0
  sum1, sum2 = 0, 0
  squares1, squares2 = 0, 0

  for x, y in pairs:
    sum1 += x
    sum2 += y
    product_sum += x * y
    squares1 += x * x
    squares2 += y * y

  size = len(pairs)
  numerator = product_sum - ((sum1 * sum2) / size)
  denominator = math.sqrt(\
    (squares1 - (sum1 * sum1) / size) *\
    (squares2 - (sum2 * sum2) / size)\
  )

  return numerator / denominator if denominator != 0 else 0


############################ CORP ##############################
def correlation_method(data: np.ndarray):
    num_features = 16
    coefs = []
    pairs = []
    for i in range(0, (num_features)):
        st1 = np.array(data[:, i])
        st2 = np.array(data[:, (num_features)])
        arr = []
        for x, y in zip(st1, st2):
            arr.append((x, y))
        cur_pearson = abs(pearson(arr))
        coefs.append(cur_pearson)
        pairs.append((i, cur_pearson))
    answers = np.zeros((128, 1))
    qwerty = data[:, :-1]
    for i in range(len(qwerty)):
        row = np.array(qwerty[i])
        cur_ans = np.dot(row, coefs)
        answers[i] = cur_ans

    data = np.concatenate((data, answers), axis=1)

    sorted_data = data[data[:, -1].argsort()[::-1]]
    pairs.sort(key = lambda x: x[1], reverse=True)

    sorted_indexes = []
    for i, _ in pairs:
        sorted_indexes.append(i)
    sorted_indexes.append((num_features))

    sorted_data = sorted_data[:, sorted_indexes]

    out_zeros = sorted_data[np.where(sorted_data[:, 16] == 0.0)]
    out_ones = sorted_data[np.where(sorted_data[:, 16] == 1.0)]
    return out_zeros[:, :-1], out_ones[:, :-1]



############################ CORS ##############################
def spearman_cor(data: np.ndarray):
    num_features = 16
    coefs = []
    pairs = []
    for i in range(0, (num_features)):
        st1 = np.array(data[:, i])
        st2 = np.array(data[:, (num_features)])

        cur_spearman = abs(spearmanr(st1, st2)[0])
        coefs.append(cur_spearman)
        pairs.append((i, cur_spearman))
    answers = np.zeros((128, 1))
    qwerty = data[:, :-1]
    for i in range(len(qwerty)):
        row = np.array(qwerty[i])
        cur_ans = np.dot(row, coefs)
        answers[i] = cur_ans

    data = np.concatenate((data, answers), axis=1)

    sorted_data = data[data[:, -1].argsort()[::-1]]
    pairs.sort(key=lambda x: x[1], reverse=True)

    sorted_indexes = []
    for i, _ in pairs:
        sorted_indexes.append(i)
    sorted_indexes.append((num_features))
    sorted_data = sorted_data[:, sorted_indexes]

    out_zeros = sorted_data[np.where(sorted_data[:, 16] == 0.0)]
    out_ones = sorted_data[np.where(sorted_data[:, 16] == 1.0)]
    return out_zeros[:, :-1], out_ones[:, :-1]


# def prepareData(data_in: np.ndarray, labels_in: np.ndarray) -> (np.ndarray, np.ndarray):
#     data_in = data_in.astype(np.float64)
#     labels_in = labels_in.astype(np.float64)
#     min_max_scaler = MinMaxScaler()
#     data_out = min_max_scaler.fit_transform(data_in)
#     min_max_scaler = MinMaxScaler()
#     labels_out = min_max_scaler.fit_transform(labels_in)
#     return data_out, labels_out


def writeData(name_in: str, zero_num: int, one_num: int, zero_in: np.ndarray, one_in: np.ndarray) -> bool:
    zero_sorted, one_sorted = zero_in, one_in#sortData(zero_in, one_in)

    zzz = np.zeros((64, 1))
    eee = np.ones((64, 1))
    zero_in = np.concatenate((zero_in, zzz), axis=1)
    one_in = np.concatenate((one_in, eee), axis=1)
    data_in = np.append(zero_in, one_in, axis=0)
    zero_out, one_out = correlation_method(data_in)
    zero_out_spearman, one_out_spearman = spearman_cor(data_in)
    zero_grid, one_grid = greedy_salesman_search(data_in)
    zero_tsp, ones_tsp = TSP(data_in.T)


    raw_path = f'{data_str_raw}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_in)
    np.save(one_raw_name, one_in)
    data_out = np.stack((zero_in, one_in))
    data_name = f'{data_str}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)

    raw_path = f'{done_data_str_raw_diag}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_sorted)
    np.save(one_raw_name, one_sorted)
    data_out = np.stack((zero_sorted, one_sorted))
    data_name = f'{done_data_str_diag}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)

    raw_path = f'{done_data_str_raw_my}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_out)
    np.save(one_raw_name, one_out)
    data_out = np.stack((zero_out, one_out))
    data_name = f'{done_data_str_my}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)

    raw_path = f'{done_data_str_raw_spearman}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_out_spearman)
    np.save(one_raw_name, one_out_spearman)
    data_out = np.stack((zero_out_spearman, one_out_spearman))
    data_name = f'{done_data_str_spearman}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)

    raw_path = f'{done_data_str_raw_my_grid}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_grid)
    np.save(one_raw_name, one_grid)
    data_out = np.stack((zero_grid, one_grid))
    data_name = f'{done_data_str_my_grid}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)

    raw_path = f'{done_data_str_raw_my_tsp}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_tsp)
    np.save(one_raw_name, ones_tsp)
    data_out = np.stack((zero_tsp, ones_tsp))
    data_name = f'{done_data_str_my_tsp}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)
    return True


def prepare(dataset_name: str, dataset_in: (np.ndarray, np.ndarray), pbar: tqdm):
    data_in, classes_in = dataset_in
    labels_in = prepareLabels(classes_in)
    (_, classes_count) = labels_in.shape
    (instances_size, features_size) = data_in.shape
    pbar.set_description("Processing %s:[%d, %d, %d]" % (dirname, instances_size, features_size, classes_count))

    if features_size < target_features:
        return False

    per_class_data = [[] for x in range(classes_count)]

    for k in range(classes_count):
        for j in range(instances_size):
            if labels_in[j][k] == 1.0:
                per_class_data[k].append(data_in[j])

    for j in range(classes_count - 1):
        for k in range(j + 1, classes_count):
            zero_data = per_class_data[j]
            zero_len = len(zero_data)
            one_data = per_class_data[k]
            one_len = len(one_data)
            if (zero_len >= target_instances) and (one_len >= target_instances):
                zero_data = np.array(zero_data)
                #print(len(zero_data))
                zero_data = np.array(zero_data[np.random.choice(zero_len, target_instances, replace=False)])
                #print(len(zero_data))
                one_data = np.array(one_data)
                one_data = np.array(one_data[np.random.choice(one_len, target_instances, replace=False)])
                scaled_zero_data, scaled_one_data = scaleData(zero_data, one_data)
                if features_size > target_features:
                    shrunk_zero_data, shrunk_one_data = shrinkFeatures(scaled_zero_data, scaled_one_data)
                else:
                    shrunk_zero_data = scaled_zero_data
                    shrunk_one_data = scaled_one_data
                writeData(dataset_name, j, k, shrunk_zero_data, shrunk_one_data)

    # if instances_size < target_size:
    #     choices = np.random.choice(instances_size, target_size, replace=True)
    #     data_in = data_in[choices]
    #     labels_in = labels_in[choices]
    #     instances_size = target_size
    #
    # labels_out = labels_in
    # if features_size + labels_size < target_size:
    #     choices = np.random.choice(instances_size, target_size, replace=False)
    #     data_out = data_in[choices]
    #     labels_out = labels_in[choices]
    #     instances_size = target_size
    #     try:
    #         data_out = makeFeatures(labels_size, data_out)
    #     except:
    #         return False
    # elif features_size + labels_size == target_size:
    #     data_out = data_in
    # else:
    #     data_out = shrinkFeatures(labels_size, data_in, labels_in)
    #
    # choices = np.random.choice(instances_size, target_size, replace=False)
    # data_to_write = data_out[choices]
    # labels_to_write = labels_out[choices]
    # result = writeData(dataset_name, data_to_write, labels_to_write)
    # return result


if __name__ == '__main__':
    target_features = 16
    target_instances = 64
    target_classes = 2
    raw_data_str = "./datasets/np_raw/"

    data_str = f"./datasets/non_normalized/dprocessed_{target_features}_{target_instances}_{target_classes}/"
    data_str_raw = f"./datasets/non_normalized/dprocessed_{target_features}_{target_instances}_{target_classes}/raw"

    done_data_str = f"./datasets/processed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw = f"./datasets/processed_{target_features}_{target_instances}_{target_classes}/raw"

    done_data_str_diag = f"./datasets/diag/dprocessed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw_diag = f"./datasets/diag/dprocessed_{target_features}_{target_instances}_{target_classes}/raw"

    done_data_str_my = f"./datasets/normalized/dprocessed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw_my = f"./datasets/normalized/dprocessed_{target_features}_{target_instances}_{target_classes}/raw"

    done_data_str_spearman = f"./datasets/spearman/dprocessed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw_spearman = f"./datasets/spearman/dprocessed_{target_features}_{target_instances}_{target_classes}/raw"

    done_data_str_my_grid = f"./datasets/normalized_grid/dprocessed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw_my_grid = f"./datasets/normalized_grid/dprocessed_{target_features}_{target_instances}_{target_classes}/raw"

    done_data_str_my_tsp = f"./datasets/normalized_tsp/dprocessed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw_my_tsp = f"./datasets/normalized_tsp/dprocessed_{target_features}_{target_instances}_{target_classes}/raw"

    data_str_path = Path(f'{data_str}')
    data_str_path.mkdir(parents=True, exist_ok=True)
    data_str_raw_path = Path(f'{data_str_raw}')
    data_str_raw_path.mkdir(parents=True, exist_ok=True)
    done_data_str_path = Path(f'{done_data_str}')
    done_data_str_path.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path = Path(f'{done_data_str_raw}')
    done_data_str_raw_path.mkdir(parents=True, exist_ok=True)

    done_data_str_path_diag = Path(f'{done_data_str_diag}')
    done_data_str_path_diag.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path_diag = Path(f'{done_data_str_raw_diag}')
    done_data_str_raw_path_diag.mkdir(parents=True, exist_ok=True)

    done_data_str_path_my = Path(f'{done_data_str_my}')
    done_data_str_path_my.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path_my = Path(f'{done_data_str_raw_my}')
    done_data_str_raw_path_my.mkdir(parents=True, exist_ok=True)

    done_data_str_path_spearman = Path(f'{done_data_str_spearman}')
    done_data_str_path_spearman.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path_spearman = Path(f'{done_data_str_raw_spearman}')
    done_data_str_raw_path_spearman.mkdir(parents=True, exist_ok=True)

    done_data_str_path_my_grid = Path(f'{done_data_str_my_grid}')
    done_data_str_path_my_grid.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path_my_grid = Path(f'{done_data_str_raw_my_grid}')
    done_data_str_raw_path_my_grid.mkdir(parents=True, exist_ok=True)

    done_data_str_path_my_tsp = Path(f'{done_data_str_my_tsp}')
    done_data_str_path_my_tsp.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path_my_tsp = Path(f'{done_data_str_raw_my_tsp}')
    done_data_str_raw_path_my_tsp.mkdir(parents=True, exist_ok=True)

    start_from = 0

    non_processed = []
    for (dirpath, dirnames, filenames) in walk(raw_data_str):
        pbar = tqdm(enumerate(dirnames), total=len(dirnames))
        for cc, dirname in pbar:
            if cc >= start_from:
                data_in, classes_in = load(dirname)
                prepare_status = prepare(dirname, load(dirname), pbar)
                if not prepare_status:
                    non_processed.append(dirname)
    print(non_processed)
