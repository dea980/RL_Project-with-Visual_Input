import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvVAE(nn.Module):
    def __init__(self, in_dim, latent_dim, device = "cpu"):
        super(ConvVAE, self).__init__()
        self.h, self.w = in_dim
        self.latent_dim = latent_dim

        # Encoder
        # (3, h, w) -> (16, h/2, w/2) -> (16, h/4, w/4)
        self.encoder_1 = nn.Conv2d(3, 8, 3, stride = 2, padding = 1)
        self.encoder_2 = nn.Conv2d(8, 16, 3, stride = 2, padding = 1)

        self.h_ = np.ceil(self.h/4).astype('int')
        self.w_ = np.ceil(self.w/4).astype('int')

        # Latent Space
        self.latent_layer_mean = nn.Linear(16 * self.h_ * self.w_, latent_dim)
        self.latent_layer_variance = nn.Linear(16 * self.h_ * self.w_, latent_dim)
        self.latent_decoder = nn.Linear(latent_dim, 16 * self.h_ * self.w_)

        # Decoder
        # (16, h/4, w/4) -> (16, h/2, w/2) -> (3, h, w)
        self.decoder_1 = nn.ConvTranspose2d(16, 8, 4, stride = 2, padding = 1)
        self.decoder_2 = nn.ConvTranspose2d(8, 3, 4, stride = 2, padding = 1)
       
        self.to(device)

    def encode(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.encoder_1(x))
        x = F.relu(self.encoder_2(x))
        x = x.reshape(batch_size, -1)
        return self.latent_layer_mean(x), self.latent_layer_variance(x)

    def reparameterization(self, mean, variance):
        std = T.exp(0.5 * variance)
        eps = T.rand_like(std)
        return mean + std * eps

    def decode(self, x):
        batch_size = x.shape[0]
        x = self.latent_decoder(x).reshape(batch_size, -1, self.h_, self.w_)
        x = F.relu(self.decoder_1(x))
        x = T.sigmoid(self.decoder_2(x))
        return x

    def forward(self, x):
        mean, variance = self.encode(x)
        z = self.reparameterization(mean, variance)
        return self.decode(z), mean, variance