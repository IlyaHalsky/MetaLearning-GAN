import os

import torch
from torch.utils import data
import numpy as np
from torchvision import transforms


class DatasetFolder(data.Dataset):

    def __init__(self, path: str):
        self.root = path
        self.data_paths = list(map(lambda x: os.path.join(self.root, x), os.listdir(self.root)))
        self.data_names = list(map(lambda x: x.split('.')[0], os.listdir(self.root)))

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data_np = np.load(data_path)
        dataset = torch.from_numpy(data_np).float()

        return (dataset, meta_features, lambda_features)

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    datasets = DatasetFolder("../processed_data/processed_50/")
    print(datasets.__getitem__(100))
