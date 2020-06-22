import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.autograd import Variable
from torch.nn import MSELoss
import numpy as np

from DatasetLoader import get_loader
from Models import Generator, Discriminator
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
import os
import math
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def getMeta(data_in: torch.Tensor, metas):
    meta_list = []
    for data in data_in:
        meta_list.append(metas.getShort(data.cpu().detach().numpy()))
    result = torch.stack(meta_list)
    return to_variable(result.view((result.size(0), result.size(1), 1, 1)))

def to_variable(x):
    return Variable(x)

exp_num = 3
datasize = 64
z_size = 100
batch_size = 1
workers = 5
lambdas = LambdaFeaturesCollector(16, 64)
metas = MetaFeaturesCollector(16, 64)
dataloader = get_loader(f"../processed_data/processed_16_64_2/", 16, 64, 2, metas, lambdas, batch_size, workers)
datatest = get_loader(f"../processed_data/test/", 16, 64, 2, metas, lambdas, batch_size, workers, train_meta=False)
discriminator = Discriminator(16, 64, 2, metas.getLength(),
                                  lambdas.getLength())
generator = Generator(16, 64, 2, metas.getLength(), 100)

methods = ['models_base', 'models_diag', 'models_corp', 'models_cors', 'models_tspg', 'models_tsph']
mse = MSELoss()

methods_results = []

for w in range(len(methods)):
    print("Method " + methods[w])
    global_reals = []
    global_fakes = []
    global_luckies = []
    for j in range(5, 55, 5):
        g_reals = []
        g_fakes = []
        g_luckies = []
        d_int_reals = []
        d_int_fakes = []
        d_int_luckies = []
        index = 0
        for i in range(exp_num):
            reals = []
            fakes = []
            luckies = []
            print("Epoch " + str(j))
            discriminator.load_state_dict(
                torch.load(
                    f'./{methods[w]}{i}/discriminator-16_64_2-{j}.pkl'))
            discriminator.eval()

            generator.load_state_dict(
                torch.load(
                    f'./{methods[w]}{i}/generator-16_64_2-{j}.pkl'))
            generator.eval()

            for k, data in enumerate(datatest):
                dataset = (data[0])
                metass = (data[1])
                lambdas = (data[2])
                batch_size = data[0].size(0)
                noise = torch.randn(batch_size, 100)
                noise = noise.view((noise.size(0), noise.size(1), 1, 1))
                noise = to_variable(noise)
                fake_data = generator(noise, metass)
                fake_data_metas = getMeta(fake_data, metas)
                fake_outputs = discriminator(fake_data, fake_data_metas)
                ones = torch.ones([len(fake_outputs), 1], dtype=torch.float32)
                d_fake_rf_loss = mse(fake_outputs[:1], ones)
                fakes.append(d_fake_rf_loss.cpu().detach().numpy())

                real_outputs = discriminator(dataset, metass)
                q = real_outputs[1:].cpu().detach().numpy()

                winners = np.argwhere(q == np.amax(q)).flatten().tolist()
                lambdas_ = lambdas.cpu().detach().numpy()
                for winner in winners:
                    if lambdas_[0][winner] == 1.0:
                        luckies[index] += 1
                d_real_labels_loss = mse(real_outputs[1:], lambdas)
                zeros = torch.zeros(len(real_outputs))
                d_real_rf_loss = mse(real_outputs[:1], zeros)
                reals.append(d_real_rf_loss.cpu().detach().numpy())
            luckies[index] /= len(datatest)
            # print(reals)
            # print(fakes)
            # print(luckies)

            g_reals.append(np.mean(reals))
            g_fakes.append((np.mean(fakes)))
            g_luckies.append(luckies[index])
            index += 1
        std_r = np.std(g_reals)
        std_f = np.std(g_fakes)
        std_l = np.std(g_luckies)
        d_int_r = 2.0 * std_r / math.sqrt(exp_num)
        d_int_f = 2.0 * std_f / math.sqrt(exp_num)
        d_int_l = 2.0 * std_l / math.sqrt(exp_num)
        d_int_reals.append(d_int_r)
        d_int_fakes.append(d_int_f)
        d_int_luckies.append(d_int_l)

        global_reals.append((np.mean(g_reals), d_int_reals))
        global_fakes.append((np.mean(g_fakes), d_int_fakes))
        global_luckies.append((np.mean(g_luckies), d_int_luckies))
    methods_results.append((global_reals, global_fakes, global_luckies))
print(methods_results)



