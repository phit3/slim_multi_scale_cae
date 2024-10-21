import torch
from torchvision.models import resnet50
from models.base_cae import BaseCAE


class HAE(torch.nn.Module):
    def __init__(self, width: int, height: int, channel_factor: int, latent_dim: int, n_stages: int):
        super().__init__()
        encoders = []
        decoders = []
        self.n_stages = n_stages
        self.stage = 0
        for i in range(n_stages):
            encoders.append(BaseCAE.Encoder(width=width, height=height, channel_factor=channel_factor, latent_dim=latent_dim))
            decoders.append(BaseCAE.Decoder(width=width, height=height, channel_factor=channel_factor, latent_dim=latent_dim * (i + 1)))
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)

    def next_stage(self):
        if self.stage < self.n_stages - 1:
            self.stage += 1
            return True
        return False

    def forward(self, inputs: torch.Tensor, mode='train') -> torch.Tensor:
        stage = self.stage if mode != 'test' else len(self.encoders) - 1
        lat_reps = []
        if stage > 0:
            for i in range(stage):
                lat_reps.append(self.encoders[i](inputs).detach().clone())
        lat_reps.append(self.encoders[stage](inputs))
        self.lat_rep = torch.cat(lat_reps, dim=1)

        return self.decoders[stage](self.lat_rep), self.lat_rep

