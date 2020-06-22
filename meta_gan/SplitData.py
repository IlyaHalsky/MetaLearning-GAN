from os import listdir
from os.path import isfile, join
import os
import shutil
from collections import defaultdict
import random

if __name__ == '__main__':
    path = f"../processed_data/dprocessed_16_64_2/"
    test_path = f"../processed_data/dtest/"
    count = 0
    onlyfiles = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    # print(onlyfiles)
    res = defaultdict(list)
    from collections import defaultdict
    for name in onlyfiles:
        key = name.split('_')[0]
        res[key].append(name)

    target = 8000
    to_keep = []

    for k in res.keys():
        size = len(res[k])
        print(size, target)
        if size <= target:
            to_keep.extend(res[k])
            target -= size

    print(to_keep)
    print(len(to_keep))
    for i, name in enumerate(onlyfiles):
        if name not in to_keep:
            os.rename(f"{path}{name}", f"{test_path}{name}")
