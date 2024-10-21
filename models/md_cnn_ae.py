import torch
from torchvision.models import resnet50
from models.base_cae import BaseCAE


class MDCNNAE(torch.nn.Module):
    def __init__(self, width: int, height: int, channel_factor: int, latent_dim: int, n_stages: int = 1):
        super().__init__()
        self.encoder = BaseCAE.Encoder(width=width, height=height, channel_factor=channel_factor, latent_dim=latent_dim)
        decoders = []
        for i in range(latent_dim):
            decoders.append(BaseCAE.Decoder(width=width, height=height, channel_factor=channel_factor, latent_dim=1))
        self.decoders = torch.nn.ModuleList(decoders)
    
    def next_stage(self):
        return False

    def forward(self, inputs: torch.Tensor, mode='train') -> torch.Tensor:
        self.lat_rep = self.encoder(inputs)

        recs = []
        for i, decoder in enumerate(self.decoders):
            recs.append(decoder(self.lat_rep[:, i: i + 1]))
        return torch.stack(recs, dim=1).sum(dim=1), self.lat_rep

