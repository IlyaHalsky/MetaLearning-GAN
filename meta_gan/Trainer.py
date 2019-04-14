import os
from datetime import datetime
from pathlib import Path

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss

from meta_gan.DatasetLoader import get_loader
from meta_gan.Models import Generator, Discriminator
from meta_gan.feature_extraction.LambdaFeatures import LambdaFeatures
from meta_gan.feature_extraction.MetaFeatures import MetaFeatures
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='train.log', level=logging.DEBUG,
                    datefmt='%d-%m %H:%M:%S')


class Trainer:
    def __init__(self, num_epochs: int = 500, cuda: bool = True):
        self.datasize = 64
        self.z_size = 100
        self.batch_size = 50
        self.workers = 5
        self.num_epochs = num_epochs
        self.cuda = cuda
        self.log_step = 9

        self.models_path = "./models"

        self.lambdas = LambdaFeatures()
        self.metas = MetaFeatures(self.datasize)
        self.data_loader = get_loader(f"../processed_data/processed_{self.datasize}/", self.datasize, self.metas,
                                      self.lambdas, self.batch_size,
                                      self.workers)

        self.generator = Generator(self.datasize, self.metas.getLength(), self.z_size)
        if self.cuda:
            self.generator.cuda()
        self.discriminator = Discriminator(self.datasize, self.metas.getLength(), self.lambdas.getLength())
        if self.cuda:
            self.discriminator.cuda()

        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.g_optimizer = optim.Adam(self.generator.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      self.lr, [self.beta1, self.beta2])

        self.cross_entropy = BCEWithLogitsLoss()
        if self.cuda:
            self.cross_entropy.cuda()

    def to_variable(self, x):
        if self.cuda:
            x = x.cuda()
        return Variable(x)

    def getMeta(self, data_in: torch.Tensor, labels_length: torch.Tensor):
        meta_list = []
        for data, m_size in zip(data_in, labels_length):
            meta_list.append(self.metas.getShort(data[0].cpu().detach().numpy(), m_size.item()))
        result = torch.stack(meta_list)
        return self.to_variable(result.view((result.size(0), result.size(1), 1, 1)))

    def getLambda(self, data_in: torch.Tensor, labels_length: torch.Tensor):
        lamba_list = []
        for data, m_size in zip(data_in, labels_length):
            lamba_list.append(self.lambdas.get(data[0].cpu().detach().numpy(), labels_num_in=m_size.item()))
        result = torch.stack(lamba_list)
        return self.to_variable(result)

    def train(self):
        total_steps = len(self.data_loader)
        logging.info(f'Starting training...')
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.data_loader):
                dataset = self.to_variable(data[0])
                metas = self.to_variable(data[1])
                lambdas = self.to_variable(data[2])
                labels_length = self.to_variable(data[3])
                batch_size = data[0].size(0)
                noise = torch.randn(batch_size, self.z_size)
                noise = noise.view((noise.size(0), noise.size(1), 1, 1))
                noise = self.to_variable(noise)
                zeros = torch.zeros([batch_size, 1], dtype=torch.float32)
                zeros = self.to_variable(zeros)
                ones = torch.ones([batch_size, 1], dtype=torch.float32)
                ones = self.to_variable(ones)

                # Get D on real
                real_outputs = self.discriminator(dataset, metas)
                d_real_labels_loss = self.cross_entropy(real_outputs[:, 1:], lambdas)
                d_real_rf_loss = self.cross_entropy(real_outputs[:, :1], zeros)
                d_real_loss = d_real_labels_loss + d_real_rf_loss

                # Get D on fake
                fake_data = self.generator(noise, metas)
                fake_outputs = self.discriminator(fake_data, metas)
                fake_lambdas = self.getLambda(fake_data, labels_length)
                d_fake_labels_loss = self.cross_entropy(fake_outputs[:, 1:], fake_lambdas)
                d_fake_rf_loss = self.cross_entropy(fake_outputs[:, :1], ones)
                d_fake_loss = d_fake_labels_loss + d_fake_rf_loss

                # Train D
                d_loss = d_real_loss + d_fake_loss
                self.discriminator.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Get D on fake
                noise = torch.randn(batch_size, self.z_size)
                noise = noise.view(noise.size(0), noise.size(1), 1, 1)
                noise = self.to_variable(noise)
                fake_data = self.generator(noise, metas)
                fake_outputs = self.discriminator(fake_data, metas)
                g_fake_rf_loss = self.cross_entropy(fake_outputs[:, :1], ones)
                fake_metas = self.getMeta(fake_data, labels_length)
                g_fake_meta_loss = self.cross_entropy(fake_metas, metas)
                g_loss = g_fake_rf_loss + g_fake_meta_loss

                # Train G
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # logging
                log = (
                    f'[{datetime.now()}] Epoch[{epoch}/{self.num_epochs}], Step[{i}/{total_steps}],'
                    f' D_losses: [{d_real_rf_loss}|{d_real_labels_loss}|{d_fake_rf_loss}|{d_fake_labels_loss}], '
                    f'G_losses:[{g_fake_rf_loss}|{g_fake_meta_loss}]'
                )
                logging.info(log)
                if (i + 1) % self.log_step == 0:
                    print(log)

            # saving
            if (epoch + 1) % 10 == 0:
                done_data_str_path = Path(self.models_path)
                done_data_str_path.mkdir(parents=True, exist_ok=True)
                g_path = os.path.join(self.models_path, 'generator-%d-%d.pkl' % (self.datasize, epoch + 1))
                d_path = os.path.join(self.models_path, 'discriminator-%d-%d.pkl' % (self.datasize, epoch + 1))
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
