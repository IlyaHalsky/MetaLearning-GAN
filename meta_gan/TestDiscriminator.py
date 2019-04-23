import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.autograd import Variable
from torch.nn import MSELoss

from meta_gan.DatasetLoader import get_loader
from meta_gan.Models import Generator, Discriminator
from meta_gan.feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from meta_gan.feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector

if __name__ == '__main__':
    datasize = 64
    z_size = 100
    batch_size = 100
    workers = 5
    lambdas = LambdaFeaturesCollector(16, 64)
    metas = MetaFeaturesCollector(16, 64)
    dataloader = get_loader(f"../processed_data/processed_16_64_2/", 16, 64, 2, metas, lambdas, batch_size, workers)
    datatest = get_loader(f"../processed_data/test/", 16, 64, 2, metas, lambdas, batch_size, workers, train_meta=False)
    discriminator = Discriminator(16, 64, 2, metas.getLength(),
                                  lambdas.getLength())
    discriminator.load_state_dict(
        torch.load(
            f'./models/discriminator-16_64_2-65.pkl'))
    discriminator.eval()
    discriminator.cuda()
    mse = MSELoss()


    def to_variable(x):
        x = x.cuda()
        return Variable(x)


    loss = []
    for i, data in enumerate(datatest):
        print(i)
        dataset = to_variable(data[0])
        metas = to_variable(data[1])
        lambdas = to_variable(data[2])
        real_outputs = discriminator(dataset, metas)
        d_real_labels_loss = mse(real_outputs[:, 1:], lambdas)
        loss.append(d_real_labels_loss.cpu().detach().numpy())

    print(loss)
