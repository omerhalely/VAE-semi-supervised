import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        mu = self.mu(x)
        var = self.var(x)

        return mu, var


class Decoder(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Linear(hidden_size, 512),
            nn.Linear(512, 32 * 7 * 7),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.size(0), 32, 7, 7))
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.encoder = Encoder(hidden_size, latent_size)
        self.decoder = Decoder(hidden_size, latent_size)

    def forward(self, x):
        mu, var = self.encoder(x)

        std = torch.exp(var / 2)
        noise = torch.randn_like(std)
        sample = mu + std * noise

        output = self.decoder(sample)
        return output, mu, std


if __name__ == "__main__":
    batch_size = 32
    c = 1
    hidden_s = 128
    latent_s = 10
    h = 28
    w = 28

    x = torch.rand(batch_size, c, h, w)

    model = VAE(hidden_s, latent_s)
    output, mu, std = model(x)
    print(f"Output Shape {output.shape} Mu Shape {mu.shape} Var Shape {std.shape}")

