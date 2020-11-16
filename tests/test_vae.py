from src.vae_will import VAE
import torch


def test_kl_divergence():
    vae = VAE(32, 1000)

    batch = 3
    channels = 3
    width = 32
    x1 = torch.rand(batch, channels, width, width)

    x1 = vae.encoder(x1)
    x1_mu, x1_logvar = vae.projection(x1)
    x1_P, x1_Q, x1_z = vae.sample(x1_mu, x1_logvar)

