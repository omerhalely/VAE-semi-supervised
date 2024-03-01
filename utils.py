import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os


def load_data(data_type, train_size):
    assert data_type in ["MNIST", "FashionMNIST"], "Data type must be MNIST/FashionMNIST."
    transform = transforms.Compose([transforms.ToTensor()])
    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        os.mkdir(os.path.join(os.getcwd(), "data"))
    if data_type == "MNIST":
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        train_dataset_labeled, train_dataset_unlabeled = torch.utils.data.random_split(train_dataset,
                                                                                       [train_size,
                                                                                        len(train_dataset) - train_size]
                                                                                       )
        test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
        return train_dataset_labeled, train_dataset_unlabeled, train_dataset, test_dataset
    if data_type == "FashionMNIST":
        train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        train_dataset_labeled, train_dataset_unlabeled = torch.utils.data.random_split(train_dataset,
                                                                                       [train_size,
                                                                                        len(train_dataset) - train_size]
                                                                                       )
        test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
        return train_dataset_labeled, train_dataset_unlabeled, train_dataset, test_dataset


def kl_divergence(mu1, std1, mu2, std2):
    kl = torch.log(std2 / std1) + ((std1 ** 2 + (mu1 - mu2) ** 2) / (2 * std2 ** 2)) - 0.5
    return kl


def save_images(model, dataset, writer, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    dataiter = iter(dataloader)

    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('Dataset Images', img_grid)

    images = images.to(device)
    output_images, mu, std = model(images)
    img_grid = torchvision.utils.make_grid(output_images)
    writer.add_image('Reconstructed images', img_grid)


def save_latent_representation(model, dataset, writer, n, batch_size, latent_size, device):
    model.eval()
    mean, var, label = get_latent_representation(model, dataset, batch_size, latent_size, device)
    mean = torch.from_numpy(mean[:n])
    var = torch.from_numpy(var[:n])
    label = label[:n].tolist()
    label = [int(label[i]) for i in range(len(label))]

    std = torch.exp(var / 2)
    noise = torch.randn_like(std)
    sample = mean + std * noise

    writer.add_embedding(sample, metadata=label)


def get_latent_representation(model, dataset, batch_size, latent_size, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    means = np.zeros((1, latent_size))
    vars = np.zeros((1, latent_size))
    labels_array = np.zeros(latent_size)

    for batch_idx, (images, labels) in enumerate(dataloader):
        with torch.no_grad():
            images = images.to(device)
            mu, var = model.encoder(images)

            mean = mu.to("cpu").numpy()
            var = var.to("cpu").numpy()
            labels = np.transpose(labels.numpy())

            labels_array = np.hstack((labels_array, labels))
            means = np.vstack((means, mean))
            vars = np.vstack((vars, var))

    return means[1:], vars[1:], labels_array[latent_size:]
