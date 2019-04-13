import os

import torch
from torch.utils import data
import numpy as np
from meta_gan.feature_extraction.MetaFeatures import MetaFeatures
from meta_gan.feature_extraction.LambdaFeatures import LambdaFeatures


class DatasetFolder(data.Dataset):

    def __init__(self, path: str, size: int, meta: MetaFeatures, lambdas: LambdaFeatures):
        self.root = path
        self.size = size
        paths = []
        for fname in os.listdir(self.root):
            path = os.path.join(self.root, fname)
            if not os.path.isdir(path):
                paths.append(path)
        self.data_paths = paths
        self.data_names = list(map(lambda x: x.split('/')[-1].split('.')[0], paths))
        self.meta_features = meta
        self.lambda_features = lambdas

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data_path = self.data_paths[index]
        data_name = self.data_names[index]
        data_np = np.load(data_path)
        dataset_tensor = torch.from_numpy(data_np).float().view(1, self.size, self.size)

        meta_tensor = self.meta_features.get(data_np, data_name) \
            .view(self.meta_features.getLength(), 1, 1)
        lambda_tensor = self.lambda_features.get(data_np, name_in=data_name) \
            .view(self.lambda_features.getLength(), 1, 1)
        return dataset_tensor, meta_tensor, lambda_tensor

    def __len__(self):
        return len(self.data_paths)


def get_loader(path: str, size:int, meta: MetaFeatures, lambdas: LambdaFeatures, batch_size: int, num_workers: int):
    datasets_inner = DatasetFolder(path, size, meta, lambdas)

    data_loader = data.DataLoader(
        dataset=datasets_inner,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return data_loader


if __name__ == '__main__':
    datasets = DatasetFolder("../processed_data/processed_50/", MetaFeatures(), LambdaFeatures())
    for i in range(len(datasets)):
        print(datasets.__getitem__(i))
