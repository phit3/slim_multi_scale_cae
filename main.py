import yaml
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
import os

from snapshot_data import SnapshotData
from cae_controller import CAEController

# check workspace
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# load config
print('Loading config...')
try:
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print(f'Could not load config: {e}')

# set seeds
seed = config['seed'] if 'seed' in config else 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# load raw data
print('Loading data...')
data_fname = config['data_params']['data_fname']
train = np.load(os.path.join('data', f'{data_fname}_train.npy'))
valid = np.load(os.path.join('data', f'{data_fname}_valid.npy'))
test = np.load(os.path.join('data', f'{data_fname}_test.npy'))

# sms-cae training
## data preparation
if 'data_params' in config and 'samples' in config['data_params']:
    samples = config['data_params']['samples']
    train_samples = int(samples * 0.8)
    valid_samples = int(samples * 0.1)
    test_samples = samples - train_samples - valid_samples
else:
    train_samples = train.shape[0]
    valid_samples = valid.shape[0]
    test_samples = test.shape[0]

train_ds = SnapshotData(train, config, subset='train', samples=train_samples)
valid_ds = SnapshotData(valid, config, subset='valid', data_min=train_ds.data_min, data_max=train_ds.data_max, samples=valid_samples)
test_ds = SnapshotData(test, config, subset='test', data_min=train_ds.data_min, data_max=train_ds.data_max, samples=test_samples)

batch_size = train_ds.batch_size
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

config['cae_params']['batch_size'] = batch_size
cae = CAEController(config)
if config['cae_params']['load_cp']:
    print('Loading checkpoint of CAE...')
    try:
        cae.load_cp()
    except Exception as e:
        print(f'Could not load checkpoint: {e}') 
        cae.train(train_dl, valid_dl)
else:
    print('Training CAE...')
    cae.train(train_dl, valid_dl)

# CAE inference
cae.infer(test_dl)
