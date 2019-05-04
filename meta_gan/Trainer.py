import os
from datetime import datetime
from pathlib import Path

import torch
from scipy.spatial.distance import mahalanobis
from torch import optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np

from meta_gan.DatasetLoader import get_loader
from meta_gan.Models import Generator, Discriminator
from meta_gan.feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from meta_gan.feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='train.log', level=logging.DEBUG,
                    datefmt='%d-%m %H:%M:%S')


class Trainer:
    def __init__(self, num_epochs: int = 500, cuda: bool = True, continue_from: int = 0):
        self.features = 16
        self.instances = 64
        self.classes = 2
        self.z_size = 100
        self.batch_size = 100
        self.workers = 5
        self.num_epochs = num_epochs
        self.cuda = cuda
        self.log_step = 10
        self.log_step_print = 50
        self.save_period = 5
        self.continue_from = continue_from

        self.models_path = "./models"

        self.lambdas = LambdaFeaturesCollector(self.features, self.instances)
        self.metas = MetaFeaturesCollector(self.features, self.instances)
        self.data_loader = get_loader(f"../processed_data/processed_{self.features}_{self.instances}_{self.classes}/",
                                      self.features, self.instances, self.classes, self.metas,
                                      self.lambdas, self.batch_size,
                                      self.workers)
        self.test_loader = get_loader(f"../processed_data/test/", 16, 64, 2, self.metas, self.lambdas, 100, 5,
                                      train_meta=False)

        if continue_from == 0:
            self.generator = Generator(self.features, self.instances, self.classes, self.metas.getLength(), self.z_size)
            self.discriminator = Discriminator(self.features, self.instances, self.classes, self.metas.getLength(),
                                               self.lambdas.getLength())
        else:
            self.generator = Generator(self.features, self.instances, self.classes, self.metas.getLength(), self.z_size)
            self.generator.load_state_dict(
                torch.load(
                    f'{self.models_path}/generator-{self.features}_{self.instances}_{self.classes}-{continue_from}.pkl'))
            self.generator.eval()

            self.discriminator = Discriminator(self.features, self.instances, self.classes, self.metas.getLength(),
                                               self.lambdas.getLength())
            self.discriminator.load_state_dict(
                torch.load(
                    f'{self.models_path}/discriminator-{self.features}_{self.instances}_{self.classes}-{continue_from}.pkl'))
            self.discriminator.eval()

        if self.cuda:
            self.generator.cuda()

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
        self.mse = MSELoss()
        if self.cuda:
            self.mse.cuda()

    def to_variable(self, x):
        if self.cuda:
            x = x.cuda()
        return Variable(x)

    def getDistance(self, x: torch.Tensor, y: torch.Tensor) -> [float]:
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

    def getMeta(self, data_in: torch.Tensor):
        meta_list = []
        for data in data_in:
            meta_list.append(self.metas.getShort(data.cpu().detach().numpy()))
        result = torch.stack(meta_list)
        return self.to_variable(result.view((result.size(0), result.size(1), 1, 1)))

    def getLambda(self, data_in: torch.Tensor):
        lamba_list = []
        for data in data_in:
            lamba_list.append(self.lambdas.get(data.cpu().detach().numpy()))
        result = torch.stack(lamba_list)
        return self.to_variable(result)

    def train(self):
        total_steps = len(self.data_loader)
        logging.info(f'Starting training...')
        for epoch in range(self.continue_from, self.num_epochs):
            results = []
            print("Starting test:")
            for i, data in enumerate(self.test_loader):
                metas = self.to_variable(data[1])
                batch_size = data[0].size(0)
                noise = torch.randn(batch_size, 100)
                noise = noise.view((noise.size(0), noise.size(1), 1, 1))
                noise = self.to_variable(noise)

                fake_data = self.generator(noise, metas)
                fake_metas = self.getMeta(fake_data)
                results.extend(self.getDistance(fake_metas, metas))
            logging.info(f'{epoch}:{np.mean(np.array(results))}')
            print(np.mean(np.array(results)))

            for i, data in enumerate(self.data_loader):
                dataset = self.to_variable(data[0])
                metas = self.to_variable(data[1])
                lambdas = self.to_variable(data[2])
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
                d_real_labels_loss = self.mse(real_outputs[:, 1:], lambdas)
                d_real_rf_loss = self.mse(real_outputs[:, :1], zeros)
                d_real_loss = d_real_labels_loss + 0.7 * d_real_rf_loss

                # Get D on fake
                fake_data = self.generator(noise, metas)
                fake_data_metas = self.getMeta(fake_data)
                fake_outputs = self.discriminator(fake_data, fake_data_metas)
                fake_lambdas = self.getLambda(fake_data)
                d_fake_labels_loss = self.cross_entropy(fake_outputs[:, 1:], fake_lambdas)
                d_fake_rf_loss = self.mse(fake_outputs[:, :1], ones)
                d_fake_loss = 0.7 * d_fake_rf_loss + 0.6 * d_fake_labels_loss

                # Train D
                d_loss = d_real_loss + 0.8 * d_fake_loss
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Get D on fake
                noise = torch.randn(batch_size, self.z_size)
                noise = noise.view(noise.size(0), noise.size(1), 1, 1)
                noise = self.to_variable(noise)
                fake_data = self.generator(noise, metas)
                fake_outputs = self.discriminator(fake_data, metas)
                g_fake_rf_loss = self.mse(fake_outputs[:, :1], zeros)
                fake_metas = self.getMeta(fake_data)
                g_fake_meta_loss = self.mse(fake_metas, metas)
                g_loss = 0.7 * g_fake_rf_loss + g_fake_meta_loss

                # Train G
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # logging
                if (i + 1) % self.log_step == 0:
                    log = (
                        f'[[{epoch},{i}],[{d_real_rf_loss},{d_real_labels_loss},{d_fake_rf_loss},{d_fake_labels_loss}],[{g_fake_rf_loss},{g_fake_meta_loss}]]'
                    )
                    logging.info(log)
                if (i + 1) % self.log_step_print == 0:
                    print((
                        f'[{datetime.now()}] Epoch[{epoch}/{self.num_epochs}], Step[{i}/{total_steps}],'
                        f' D_losses: [{d_real_rf_loss}|{d_real_labels_loss}|{d_fake_rf_loss}|{d_fake_labels_loss}], '
                        f'G_losses:[{g_fake_rf_loss}|{g_fake_meta_loss}]'
                    ))

            # saving
            if (epoch + 1) % self.save_period == 0:
                done_data_str_path = Path(self.models_path)
                done_data_str_path.mkdir(parents=True, exist_ok=True)
                g_path = os.path.join(self.models_path,
                                      f'generator-{self.features}_{self.instances}_{self.classes}-{epoch + 1}.pkl')
                d_path = os.path.join(self.models_path,
                                      f'discriminator-{self.features}_{self.instances}_{self.classes}-{epoch + 1}.pkl')
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
