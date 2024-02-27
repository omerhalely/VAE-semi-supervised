import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, height, width, hidden_size, latent_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 24 * 24, 512)
        self.fc2 = nn.Linear(512, hidden_size)

        self.mu = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        var = self.var(x)

        return mu, var


class Decoder(nn.Module):
    def __init__(self, height, width, hidden_size, latent_size):
        super().__init__()
        self.height = height
        self.width = width
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 512)
        self.fc3 = nn.Linear(512, 32 * 24 * 24)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3)
        self.upconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.reshape(x, (x.size(0), 32, 24, 24))

        x = F.relu(self.upconv1(x))
        x = F.sigmoid(self.upconv2(x))
        return x


class VAE(nn.Module):
    def __init__(self, height, width, hidden_size, latent_size):
        super().__init__()

        self.encoder = Encoder(height, width, hidden_size, latent_size)
        self.decoder = Decoder(height, width, hidden_size, latent_size)

    def forward(self, x):
        z_mu, z_var = self.encoder(x)

        std = torch.exp(z_var)
        epsilon = torch.randn_like(std)
        sample = z_mu + std * epsilon

        output_image = self.decoder(sample)

        return output_image, z_mu, z_var


if __name__ == "__main__":
    batch_size = 32
    c = 1
    h = 28
    w = 28
    hidden_s = 128
    latent_s = 10

    x = torch.rand(batch_size, c, h, w)

    model = VAE(height=h, width=w, hidden_size=hidden_s, latent_size=latent_s)
    output, mu, var = model(x)

    print(f"Output Shape {output.shape} Mu Shape {mu.shape} Var Shape {var.shape}")
