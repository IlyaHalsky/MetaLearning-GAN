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

from meta_gan.feature_extraction.MetaZerosCollector import MetaZerosCollector

logging.basicConfig(format='%(asctime)s %(message)s', filename='1206_cnn_no_meta.log', level=logging.DEBUG,
                    datefmt='%d-%m %H:%M:%S')


class TrainerCNN:
    def __init__(self, num_epochs: int = 20, cuda: bool = True, continue_from: int = 0):
        self.features = 16
        self.instances = 64
        self.classes = 2
        self.z_size = 100
        self.batch_size = 100
        self.workers = 4
        self.num_epochs = num_epochs
        self.cuda = cuda
        self.log_step = 10
        self.log_step_print = 50
        self.save_period = 5
        self.continue_from = continue_from

        self.models_path = "./cnn1206"

        self.lambdas = LambdaFeaturesCollector(self.features, self.instances)
        self.metas = MetaZerosCollector(self.features, self.instances)
        self.data_loader = get_loader(f"../processed_data/processed_{self.features}_{self.instances}_{self.classes}/",
                                      self.features, self.instances, self.classes, self.metas,
                                      self.lambdas, self.batch_size,
                                      self.workers)
        self.test_loader = get_loader(f"../processed_data/test/", 16, 64, 2, self.metas, self.lambdas, 147,
                                      self.workers,
                                      train_meta=False)

        if continue_from == 0:
            self.discriminator = Discriminator(self.features, self.instances, self.classes, self.metas.getLength(),
                                               self.lambdas.getLength())
        else:
            self.discriminator = Discriminator(self.features, self.instances, self.classes, self.metas.getLength(),
                                               self.lambdas.getLength())
            self.discriminator.load_state_dict(
                torch.load(
                    f'{self.models_path}/discriminator-{self.features}_{self.instances}_{self.classes}-{continue_from}.pkl'))
            self.discriminator.eval()

        if self.cuda:
            self.discriminator.cuda()

        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

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

    def train(self):
        total_steps = len(self.data_loader)
        logging.info(f'Starting training...')
        for epoch in range(self.continue_from, self.num_epochs):
            loss = []
            for i, data in enumerate(self.test_loader):
                dataset = self.to_variable(data[0])
                metas = self.to_variable(data[1])
                lambdas = self.to_variable(data[2])
                real_outputs = self.discriminator(dataset, metas)
                d_real_labels_loss = self.mse(real_outputs[:, 1:], lambdas)
                loss.append(d_real_labels_loss.cpu().detach().numpy())
            logging.info(f'{epoch}d:{np.mean(loss)}')

            for i, data in enumerate(self.data_loader):
                dataset = self.to_variable(data[0])
                metas = self.to_variable(data[1])
                lambdas = self.to_variable(data[2])

                # Get D on real
                real_outputs = self.discriminator(dataset, metas)
                d_real_labels_loss = self.mse(real_outputs[:, 1:], lambdas)
                d_real_loss = d_real_labels_loss

                # Train D
                d_loss = d_real_loss
                self.discriminator.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # # logging
                # if (i + 1) % self.log_step == 0:
                #     log = (
                #         f'[[{epoch},{i}],[{d_real_labels_loss}]'
                #     )
                #     logging.info(log)
                if (i + 1) % self.log_step_print == 0:
                    print((
                        f'[{datetime.now()}] Epoch[{epoch}/{self.num_epochs}], Step[{i}/{total_steps}],'
                        f' D_losses: [{d_real_labels_loss}], '
                    ))

            # saving
            if (epoch + 1) % self.save_period == 0:
                done_data_str_path = Path(self.models_path)
                done_data_str_path.mkdir(parents=True, exist_ok=True)
                d_path = os.path.join(self.models_path,
                                      f'discriminator-{self.features}_{self.instances}_{self.classes}-{epoch + 1}.pkl')
                torch.save(self.discriminator.state_dict(), d_path)


if __name__ == '__main__':
    trainer = TrainerCNN()
    trainer.train()
    import winsound

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    print("123")
