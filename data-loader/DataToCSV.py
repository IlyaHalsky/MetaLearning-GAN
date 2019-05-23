from pathlib import Path

import numpy as np
from os import listdir
from os.path import isfile, join
import csv

from tqdm import tqdm

if __name__ == '__main__':
    csv_name_path = "./datasets/processed_16_64_2"
    csv_path = "./csv/16_64_2/"
    csv_path_path = Path(f'{csv_path}')
    csv_path_path.mkdir(parents=True, exist_ok=True)
    only_files = [f for f in listdir(csv_name_path) if isfile(join(csv_name_path, f))]
    for name in tqdm(only_files, total=len(only_files)):
        data = np.load(f'{csv_name_path}/{name}')
        csv_name = name.split('.')[0]
        csv_name_folder = f'{csv_path}{csv_name}/'
        csv_save_path = Path(csv_name_folder)
        csv_save_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'{csv_name_folder}zero.csv', data[0], delimiter=",")
        np.savetxt(f'{csv_name_folder}one.csv', data[1], delimiter=",")
