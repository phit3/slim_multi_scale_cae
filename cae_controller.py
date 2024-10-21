import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import random

from models.base_cae import BaseCAE
from models.sms_cae import SMSCAE
from models.h_ae import HAE
from models.md_cnn_ae import MDCNNAE

class CAEController:
    @property
    def learning_rate(self):
        if 'cae_params' in self.config:
            if 'learning_rate' in self.config['cae_params']:
                return self.config['cae_params']['learning_rate']
        return 1.0e-4

    @property
    def epochs(self):
        if 'cae_params' in self.config:
            if 'epochs' in self.config['cae_params']:
                return self.config['cae_params']['epochs']
        return 10000

    @property
    def max_patience(self):
        if 'cae_params' in self.config:
            if 'patience' in self.config['cae_params']:
                return self.config['cae_params']['patience']
        return 30

    @property
    def width(self):
        if 'cae_params' in self.config:
            if 'width' in self.config['cae_params']:
                return self.config['cae_params']['width']
        return 64

    @property
    def height(self):
        if 'cae_params' in self.config:
            if 'height' in self.config['cae_params']:
                return self.config['cae_params']['height']
        return 64

    @property
    def channel_factor(self):
        if 'cae_params' in self.config:
            if 'channel_factor' in self.config['cae_params']:
                return self.config['cae_params']['channel_factor']
        return 32

    @property
    def latent_dim(self):
        if 'cae_params' in self.config:
            if 'latent_dim' in self.config['cae_params']:
                return self.config['cae_params']['latent_dim']
        return 256

    @property
    def n_stages(self):
        if 'cae_params' in self.config:
            if 'n_stages' in self.config['cae_params']:
                return self.config['cae_params']['n_stages']
        return 1

    @property
    def cp_fname(self):
        if 'cae_params' in self.config:
            if 'cp_fname' in self.config['cae_params']:
                return self.config['cae_params']['cp_fname']
        return 'test_cp'
    

    def __init__(self, config):
        self.config = config
        self.cae_class = eval(config['cae_params']['cae_class'])
        self.cae = self.cae_class(self.width, self.height, self.channel_factor, self.latent_dim, self.n_stages).cuda()
        self.optimizer = optim.Adam(self.cae.parameters(), lr=self.learning_rate)
        self.patience = self.max_patience

    def do_batch(self, batch, mode='train'):
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        predictions, _ = self.cae(inputs, mode=mode)
        loss = F.mse_loss(predictions, targets)
        return predictions, targets, loss

    def train(self, train_dl, valid_dl):
        self.best_vloss = np.inf
        for epoch in range(self.epochs):
            # train phase
            tlosses = []
            print(f'epoch {epoch + 1}/{self.epochs}')
            self.cae.train() 
            for i, tbatch in enumerate(train_dl):
                self.optimizer.zero_grad()
                _, _, tloss = self.do_batch(tbatch, mode='train')
                tloss.backward()
                self.optimizer.step()
                tlosses.append(tloss.item())
                print(f'\rtraining {i + 1}/{len(train_dl)} - loss: {np.mean(tlosses):.5}       ', end='', flush=True)
            print()
            
            # valid phase
            vlosses = []
            self.cae.eval()
            with torch.no_grad():
                for i, vbatch in enumerate(valid_dl):
                    _, _, vloss = self.do_batch(vbatch, mode='valid')
                    vlosses.append(vloss.item())
                    print(f'\rvalidation {i + 1}/{len(valid_dl)} - loss: {np.mean(vlosses):.5}     ', end='', flush=True)
            print()
            epoch_vloss = np.mean(vlosses)

            # early stopping
            if epoch_vloss < self.best_vloss:
                self.patience = self.max_patience
                # save model
                self.best_vloss = epoch_vloss
                torch.save(self.cae.state_dict(), os.path.join('checkpoints', f'{self.cp_fname}.pth'))
                print('saved model')
            elif self.patience > 0:
                self.patience -= 1
            else:
                if self.cae.next_stage():
                    self.patience = self.max_patience
                    self.best_vloss = np.inf
                    print('next stage')
                else:
                    print('early stopping')
                    break

    def load_cp(self):
        self.cae.load_state_dict(torch.load(os.path.join('checkpoints', f'{self.cp_fname}.pth')))
    
    def infer(self, test_dl):
        input_data = []
        rec_data = []
        with torch.no_grad():
            self.cae.eval()
            for i, test_batch in enumerate(test_dl):
                inputs, targets = test_batch
                print(f'\rInfering {i + 1}/{len(test_dl)}', end='', flush=True)
                recs, _ = self.cae(inputs.cuda(), mode='test')
                input_data.append(inputs)
                rec_data.append(recs)
            print()
        input_data = torch.cat(input_data, dim=0)
        rec_data = torch.cat(rec_data, dim=0)
        return input_data, rec_data
