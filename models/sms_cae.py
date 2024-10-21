import torch
from torchvision.models import resnet50
from models.base_cae import BaseCAE


class SMSCAE(torch.nn.Module):
    def __init__(self, width: int, height: int, channel_factor: int, latent_dim: int, n_stages: int = 1):
        super().__init__()
        self.n_stages = n_stages
        self.encoder = BaseCAE.Encoder(width=width, height=height, channel_factor=channel_factor, latent_dim=latent_dim)
        self.decoder = BaseCAE.Decoder(width=width, height=height, channel_factor=channel_factor, latent_dim=latent_dim)
    
    def next_stage(self):
        return False

    def forward(self, inputs: torch.Tensor, mode='train') -> torch.Tensor:
        self.lat_rep = self.encoder(inputs)
        dims = self.lat_rep.shape[1]

        if mode != 'train':
            idxs = (torch.ones(self.lat_rep.shape[0]) * dims).int()
        else:
            idxs = torch.randint(1, dims + 1, (self.lat_rep.shape[0],))

        mask = torch.ones_like(self.lat_rep).float()
        for i, idx in enumerate(idxs):
            mask[i, idx:] = 0.0
        self.lat_rep = self.lat_rep * mask
        return self.decoder(self.lat_rep), self.lat_rep

