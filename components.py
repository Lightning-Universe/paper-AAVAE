from torch import nn


def _conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def _deconv(in_channels, out_channels, non_linearity=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        ),
        nn.BatchNorm2d(out_channels),
        non_linearity(),
    )


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, in_channels=3, max_channels=128):
        super().__init__()
        # construct encoder
        self.encoder = nn.Sequential(
            _conv(in_channels, max_channels // 4),
            _conv(max_channels // 4, max_channels // 2),
            _conv(max_channels // 2, max_channels),
        )
        self.fc_mu = nn.Linear(max_channels * 16, latent_dim)
        self.fc_var = nn.Linear(max_channels * 16, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, max_channels=128):
        super().__init__()
        # construct decoder
        self.max_channels = max_channels
        self.decoder_in = nn.Linear(latent_dim, max_channels * 16)
        self.decoder = nn.Sequential(
            _deconv(max_channels, max_channels // 2),
            _deconv(max_channels // 2, max_channels // 4),
            _deconv(max_channels // 4, out_channels, non_linearity=nn.Sigmoid),
        )

    def forward(self, z):
        x_hat = self.decoder_in(z)
        x_hat = x_hat.view(-1, self.max_channels, 4, 4)
        return self.decoder(x_hat)
