import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from meta_gan.DatasetLoader import get_loader
from meta_gan.Models import Generator, Discriminator
from meta_gan.feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from meta_gan.feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector

if __name__ == '__main__':
    datasize = 64
    z_size = 100
    batch_size = 1
    workers = 5
    lambdas = LambdaFeaturesCollector()
    metas = MetaFeaturesCollector(datasize)
    dataloader = get_loader(f"../processed_data/processed_{datasize}/", datasize, metas, lambdas, batch_size, workers)

    generator = Generator(datasize, metas.getLength(), z_size)
    discriminator = Discriminator(datasize, metas.getLength(), lambdas.getLength())

    meta_list = []
    lambdas_list = []
    for i, (data, meta, lambda_l, l) in enumerate(dataloader):
        print(i)
        meta_o = meta[:, :].numpy()
        meta_o = meta_o.ravel()
        meta_o = meta_o.tolist()
        meta_list.append(meta_o)
        lambdas_o = lambda_l[:, :].numpy().astype(int).ravel().tolist()
        lambdas_list.append(lambdas_o)
        '''noise = torch.randn(batch_size, z_size)
        noise = noise.view(noise.size(0), noise.size(1), 1, 1)

        g_outputs = generator(noise, meta)
        print(g_outputs)
        d_outputs = discriminator(g_outputs, meta)
        print(d_outputs)
        d_outputs = discriminator(data, meta)
        print(d_outputs)
        break'''

    split = 700
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(meta_list[:split], lambdas_list[:split])
    pred = dt.predict(meta_list[split:])
    score = mean_squared_error(pred, lambdas_list[split:])
    print(score)

