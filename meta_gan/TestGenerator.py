import torch
from scipy.spatial.distance import mahalanobis
from torch.autograd import Variable

from DatasetLoader import get_loader
from Models import Generator
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
import numpy as np
import os
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error
import math

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def to_variable(x):
    #x = x.cuda()
    return Variable(x)


def getMeta(data_in: torch.Tensor):
    meta_list = []
    for data in data_in:
        meta_list.append(metaCollector.getShort(data.cpu().detach().numpy()))
    result = torch.stack(meta_list)
    return to_variable(result.view((result.size(0), result.size(1), 1, 1)))


def getDistance(x: torch.Tensor, y: torch.Tensor) -> [float]:
    x_in = np.squeeze(x.cpu().detach().numpy())
    y_in = np.squeeze(y.cpu().detach().numpy())
    results = []
    for (xx, yy) in zip(x_in, y_in):
        try:
            V = np.cov(np.array([xx, yy]).T)
            V[np.diag_indices_from(V)] += 0.1
            IV = np.linalg.inv(V)
            D = mahalanobis(xx, yy, IV)
        except:
            D = 0.0
        results.append(D)
    return results


if __name__ == '__main__':
    exp_num = 3
    metaCollector = MetaFeaturesCollector(16, 64)
    metaCollector.train(f"../processed_data/processed_16_64_2/")
    lambdas = LambdaFeaturesCollector(16, 64)
    loader = get_loader(f"../processed_data/test/", 16, 64, 2, metaCollector, lambdas, 100, 5, train_meta=False)
    generator = Generator(16, 64, 2, metaCollector.getLength(), 100)
    methods = ['models_base', 'models_diag', 'models_corp', 'models_cors', 'models_tspg', 'models_tsph']
    methods_results = []
    for w in range(len(methods)):
        print("Method " + methods[w])
        epoch_results = []
        for j in range(5, 55, 5):
            meta_results = []
            for i in range(exp_num):
                generator.load_state_dict(
                    torch.load(
                        f'./{methods[w]}{i}/generator-16_64_2-{j}.pkl'))
                generator.eval()

                results = []
                mse = MSELoss()
                for i, data in enumerate(loader):
                    metas = to_variable(data[1])
                    batch_size = data[0].size(0)
                    noise = torch.randn(batch_size, 100)
                    noise = noise.view((noise.size(0), noise.size(1), 1, 1))
                    noise = to_variable(noise)

                    fake_data = generator(noise, metas)
                    fake_metas = getMeta(fake_data)
                    x_in = np.squeeze(metas.cpu().detach().numpy())
                    y_in = np.squeeze(fake_metas.cpu().detach().numpy())
                    for x, y in zip(x_in, y_in):
                        results.append(mean_squared_error(x, y))

                std = np.std(results)
                d_int = 2.0 * std / math.sqrt(exp_num)
                meta_results.append((np.mean(results), d_int))
            epoch_results.append(meta_results)
        methods_results.append(epoch_results)
    print(methods_results)
