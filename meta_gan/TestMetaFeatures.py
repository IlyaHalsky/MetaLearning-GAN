import os
import numpy as np
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm

from meta_gan.feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector

if __name__ == '__main__':
    metaCollector = MetaFeaturesCollector(16, 64)
    root = f"../processed_data/processed_16_64_2/"
    metaCollector.train(f"../processed_data/processed_16_64_2/")
    paths = []

    for fname in os.listdir(root):
        path = os.path.join(root, fname)
        if not os.path.isdir(path):
            paths.append(path)

    lengths = []
    metas = []
    for i in tqdm(paths):
        x = np.load(i)
        x_meta = metaCollector.getNumpy(x)
        metas.append(x_meta)

    for x_meta in tqdm(metas):
        local = []
        for y_meta in metas:
            V = np.cov(np.array([x_meta, y_meta]).T)
            V[np.diag_indices_from(V)] += 0.1
            IV = np.linalg.inv(V)
            D = mahalanobis(x_meta, y_meta, IV)
            local.append(D)
        lengths.append(np.mean(np.array(local)))
        print(np.mean(np.array(lengths)))
    print(np.mean(np.array(lengths)))
