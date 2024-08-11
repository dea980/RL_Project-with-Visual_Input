import torch as T
import torch.nn as nn
import torch.nn.functional as F

class LinearVAE(nn.Module):
    def __init__(self, in_dim, latent_dim, device = "cpu"):
        super(LinearVAE, self).__init__()
        self.h, self.w = in_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_1 = nn.Linear(self.h * self.w, 1024)
        self.encoder_2 = nn.Linear(1024, 1024)

        # Latent Space
        self.latent_layer_mean = nn.Linear(1024, latent_dim)
        self.latent_layer_variance = nn.Linear(1024, latent_dim)
        self.latent_decoder = nn.Linear(latent_dim, 1024)

        # Decoder
        self.decoder_1 = nn.Linear(1024, 1024)
        self.decoder_2 = nn.Linear(1024, self.h * self.w)
       
        self.to(device)

    def encode(self, x):
        x = F.relu(self.encoder_1(x))
        x = F.relu(self.encoder_2(x))
        x = x.reshape(-1, 1024)
        return self.latent_layer_mean(x), self.latent_layer_variance(x)

    def reparameterization(self, mean, variance):
        std = T.exp(0.5 * variance)
        eps = T.rand_like(std)
        return mean + std * eps

    def decode(self, x):
        x = self.latent_decoder(x).reshape(-1, 1024)
        x = F.relu(self.decoder_1(x))
        x = T.sigmoid(self.decoder_2(x))
        return x

    def forward(self, x):
        mean, variance = self.encode(x)
        z = self.reparameterization(mean, variance)
        return self.decode(z), mean, variance