import os

import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from meta_gan.feature_extraction.MetaFeatures import MetaFeatures
from meta_gan.feature_extraction.LambdaFeatures import LambdaFeatures


class DatasetFolder(data.Dataset):

    def __init__(self, path: str):
        self.root = path
        self.data_paths = list(map(lambda x: os.path.join(self.root, x), os.listdir(self.root)))
        self.data_names = list(map(lambda x: x.split('.')[0], os.listdir(self.root)))
        self.meta_features = MetaFeatures()
        self.lambda_features = LambdaFeatures()

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data_path = self.data_paths[index]
        data_name = self.data_names[index]
        data_np = np.load(data_path)
        dataset = torch.from_numpy(data_np).float()

        metas = self.meta_features.get(data_np, data_name)

        lambdas = self.lambda_features.get(data_np, name_in=data_name)
        lambda_tensor = torch.from_numpy(lambdas).float()
        return dataset, metas, lambda_tensor

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    datasets = DatasetFolder("../processed_data/processed_50/")
    print(datasets.__getitem__(100))
