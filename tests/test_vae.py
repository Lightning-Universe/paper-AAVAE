from src.vae_will import VAE
import torch
import pytorch_lightning as pl


def test_kl_divergence():
    pl.seed_everything(1234)
    vae = VAE(input_height=32,
              num_train_samples=1000,
              first_conv=True,
              maxpool1=True,
              encoder='resnet50',
              num_mc_samples=1000)

    batch = 3
    channels = 3
    width = 32
    x1 = torch.rand(batch, channels, width, width)

    x1 = vae.encoder(x1)
    x1_mu, x1_logvar = vae.projection(x1)
    x1_P, x1_Q, x1_z = vae.sample(x1_mu, x1_logvar)

    kl = vae.kl_divergence_mc(x1_P, x1_Q, x1_z).sum()
    pt_kl = torch.distributions.kl.kl_divergence(x1_P, x1_Q).sum(-1).mean(-1).sum()
    analytical = -0.5 * torch.sum(1 + x1_logvar - x1_mu.pow(2) - x1_logvar.exp())

    pt_deltas = (kl - pt_kl).abs()
    pt_analytical = (pt_kl - analytical).abs()
    analytical_deltas = (kl - analytical).abs()

    assert analytical_deltas < 1.0


if __name__ == '__main__':
    test_kl_divergence()
