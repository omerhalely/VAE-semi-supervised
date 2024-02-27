import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


torch.manual_seed(10)


class Handler:
    def __init__(self, model_name, data_type, height, width, hidden_size, latent_size, lr, epochs, batch_size,
                 train_size, device):
        self.model_name = model_name
        self.data_type = data_type
        self.height = height
        self.width = width
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_size = train_size
        self.device = device
        self.train_dataset, self.test_dataset = self.load_data()
        self.model = VAE(height=height, width=width, hidden_size=hidden_size, latent_size=latent_size)
        self.model.to(self.device)
        # self.model.apply(self.init_xavier)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    def init_xavier(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        if not os.path.exists(os.path.join(os.getcwd(), "data")):
            os.mkdir(os.path.join(os.getcwd(), "data"))
        if self.data_type == "MNIST":
            train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
            train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                             [self.train_size,len(train_dataset) - self.train_size])
            test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
            return train_dataset, test_dataset
        if self.data_type == "FashionMNIST":
            train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
            train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                             [self.train_size, len(train_dataset) - self.train_size])
            test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
            return train_dataset, test_dataset

    def train_one_epoch(self, epoch):
        self.model.train()

        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        bce_loss = nn.BCELoss(reduction="sum")

        total_loss = 0
        average_loss = 0
        total_kl_loss = 0
        total_reconstruction_loss = 0

        print_every = len(dataloader) // min(len(dataloader), 10)
        for batch_idx, (x, _) in enumerate(dataloader):
            if batch_idx % print_every == 0 and batch_idx != 0:
                print(f"| Epoch {epoch + 1} | Loss {average_loss:.2f} "
                      f"| KL Loss {(total_kl_loss / (len(x) * batch_idx)):.2f} "
                      f"| Reconstruction Loss {(total_reconstruction_loss / (len(x) * batch_idx)):.2f} |")

            x = x.to(self.device)

            output, mu, var = self.model(x)

            self.optimizer.zero_grad()

            reconstruction_loss = bce_loss(output, x)
            kl_loss = 0.5 * torch.sum((torch.exp(var) - 1) ** 2 + mu ** 2)

            loss = reconstruction_loss + kl_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            average_loss = total_loss / (len(x) * (batch_idx + 1))

        return average_loss

    def evaluate_model_on_dataset(self, dataset):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        bce_loss = nn.BCELoss(reduction="sum")

        total_loss = 0

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(self.device)

            with torch.no_grad():
                output, mu, var = self.model(x)

                reconstruction_loss = bce_loss(output, x)
                kl_loss = 0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1.0 - var)
                # kl_loss = 0.5 * torch.sum((torch.exp(var) - 1) ** 2 + mu ** 2)

                loss = reconstruction_loss + kl_loss

                total_loss += loss.item()
        return total_loss / len(dataset)

    def run(self):
        print(f"Start Running {self.model_name}")
        if not os.path.exists(os.path.join(os.getcwd(), "models")):
            os.mkdir(os.path.join(os.getcwd(), "models"))
        if not os.path.exists(os.path.join(os.getcwd(), "models", self.model_name)):
            os.mkdir(os.path.join(os.getcwd(), "models", self.model_name))

        checkpoint_filename = os.path.join(os.getcwd(), "models", self.model_name, self.model_name + ".pt")

        best_loss = torch.inf
        for epoch in range(self.epochs):
            print("-" * 70)
            self.train_one_epoch(epoch)

            train_loss = self.evaluate_model_on_dataset(self.train_dataset)
            test_loss = self.evaluate_model_on_dataset(self.test_dataset)

            print("-" * 70)
            print(f"| End of epoch {epoch + 1} | Train Loss {train_loss:.2f} | Test Loss {test_loss:.2f} |")

            if train_loss < best_loss:
                state = {
                    "model": self.model.state_dict()
                }
                torch.save(state, checkpoint_filename)
                best_loss = train_loss

        print("-" * 70)


if __name__ == "__main__":
    model_name = "VAE"
    data_type = "MNIST"
    height = 28
    width = 28
    hidden_size = 256
    latent_size = 10
    lr = 0.001
    epochs = 30
    batch_size = 10
    train_size = 100
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    handler = Handler(model_name=model_name,
                      data_type=data_type,
                      height=height,
                      width=width,
                      hidden_size=hidden_size,
                      latent_size=latent_size,
                      lr=lr,
                      epochs=epochs,
                      batch_size=batch_size,
                      train_size=train_size,
                      device=device)

    handler.run()

