import torch

from meta_gan.DatasetLoader import get_loader
from meta_gan.Models import Generator, Discriminator
from meta_gan.feature_extraction.LambdaFeatures import LambdaFeatures
from meta_gan.feature_extraction.MetaFeatures import MetaFeatures

if __name__ == '__main__':
    datasize = 64
    z_size = 100
    batch_size = 100
    workers = 2
    lambdas = LambdaFeatures()
    metas = MetaFeatures(datasize)
    dataloader = get_loader(f"../processed_data/processed_{datasize}/", datasize, metas, lambdas, batch_size, workers)

    generator = Generator(datasize, metas.getLength(), z_size)
    discriminator = Discriminator(datasize, metas.getLength(), lambdas.getLength())

    for i, (data, meta, lambda_l) in enumerate(dataloader):
        noise = torch.randn(batch_size, z_size)
        noise = noise.view(noise.size(0), noise.size(1), 1, 1)

        g_outputs = generator(noise, meta)
        print(g_outputs)
        d_outputs = discriminator(g_outputs, meta)
        print(d_outputs)
        d_outputs = discriminator(data, meta)
        print(d_outputs)
        break
