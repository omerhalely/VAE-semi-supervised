import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import joblib
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import VAE
from utils import load_data, kl_divergence, save_images, save_latent_representation, get_latent_representation
from sklearn.svm import SVC


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
        self.train_dataset, self.test_dataset = load_data(self.data_type, self.train_size)
        print("Building Model...")
        self.model = VAE(hidden_size=hidden_size, latent_size=latent_size)
        self.model.to(self.device)
        self.model.apply(self.init_xavier)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.writer = SummaryWriter(f"runs/{self.model_name}")

    def init_xavier(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

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

            output, mu, std = self.model(x)

            self.optimizer.zero_grad()

            reconstruction_loss = bce_loss(output, x)
            kl_loss = torch.sum(kl_divergence(mu, std, 0, 1))

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
                output, mu, std = self.model(x)

                reconstruction_loss = bce_loss(output, x)
                kl_loss = torch.sum(kl_divergence(mu, std, 0, 1))

                loss = reconstruction_loss + kl_loss

                total_loss += loss.item()
        return total_loss / len(dataset)

    def load_model(self):
        print(f"Loading model {self.model_name}.")
        model_path = os.path.join(os.getcwd(), "models", self.model_name, f"{self.model_name}.pt")
        assert os.path.exists(model_path), f"Model {self.model_name}.pt does not exist."

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        print("Loaded Model Successfully.")

    def train_model(self):
        self.model.train()
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

            self.writer.add_scalars(f"Loss/{self.model_name}", {"Train": train_loss, "Test": test_loss}, epoch)
            # self.writer.add_scalar(f"Loss/{self.model_name}", train_loss, epoch)
            # self.writer.add_scalar(f"Loss/{self.model_name}", test_loss, epoch)

            print("-" * 70)
            print(f"| End of epoch {epoch + 1} | Train Loss {train_loss:.2f} | Test Loss {test_loss:.2f} |")

            if train_loss < best_loss:
                print(f"Saving model to {checkpoint_filename}")
                state = {
                    "model": self.model.state_dict()
                }
                torch.save(state, checkpoint_filename)
                best_loss = train_loss

        print("-" * 70)
        save_images(self.model, self.train_dataset, self.writer, self.device)
        save_latent_representation(self.model, self.train_dataset, self.writer, 100, self.batch_size, self.latent_size,
                                   self.device)

    def train_classifier(self, x, y):
        print("Training linear SVM classifier.")
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(x, y)
        return svm_classifier

    def save_classifier(self, classifier):
        classifier_path = os.path.join(os.getcwd(), "models", self.model_name, f"{self.model_name}.pkl")
        print(f"Saving classifier to {classifier_path}.")
        joblib.dump(classifier, classifier_path)

    def load_classifier(self):
        classifier_path = os.path.join(os.getcwd(), "models", self.model_name, f"{self.model_name}.pkl")
        assert os.path.exists(classifier_path), f"Classifier {model_name}.pkl does not exist."
        print(f"Loading classifier from {classifier_path}")
        classifier = joblib.load(classifier_path)
        return classifier

    def train(self):
        print(f"Running on {self.device}")
        self.train_model()
        self.load_model()

        train_means, train_vars, train_labels = get_latent_representation(self.model, self.train_dataset,
                                                                          self.batch_size, self.latent_size,
                                                                          self.device)
        train_data = np.hstack((train_means, train_vars))
        svm_classifier = self.train_classifier(train_data, train_labels)

        train_predictions = svm_classifier.predict(train_data)
        train_accuracy = np.sum(train_predictions == train_labels) / len(train_predictions)

        test_means, test_vars, test_labels = get_latent_representation(self.model, self.test_dataset, self.batch_size,
                                                                       self.latent_size, self.device)
        test_data = np.hstack((test_means, test_vars))
        test_predictions = svm_classifier.predict(test_data)
        test_accuracy = np.sum(test_predictions == test_labels) / len(test_predictions)

        print(f"| Train Classification Accuracy {train_accuracy * 100:.2f}% | Test Classification Accuracy "
              f"{test_accuracy * 100:.2f}% |")

        self.save_classifier(svm_classifier)

    def test(self):
        print(f"Testing model {self.model_name}")
        self.load_model()
        svm_classifier = self.load_classifier()

        train_means, train_vars, train_labels = get_latent_representation(self.model, self.train_dataset,
                                                                          self.batch_size, self.latent_size,
                                                                          self.device)
        test_means, test_vars, test_labels = get_latent_representation(self.model, self.test_dataset, self.batch_size,
                                                                       self.latent_size, self.device)
        train_data = np.hstack((train_means, train_vars))
        test_data = np.hstack((test_means, test_vars))

        train_predictions = svm_classifier.predict(train_data)
        train_accuracy = np.sum(train_predictions == train_labels) / len(train_predictions)
        test_predictions = svm_classifier.predict(test_data)
        test_accuracy = np.sum(test_predictions == test_labels) / len(test_predictions)

        print(f"| Train Classification Accuracy {train_accuracy * 100:.2f}% | Test Classification Accuracy "
              f"{test_accuracy * 100:.2f}% |")


if __name__ == "__main__":
    train_size = 100
    model_name = f"VAE_{train_size}"
    data_type = "MNIST"
    height = 28
    width = 28
    hidden_size = 256
    latent_size = 10
    lr = 0.001
    epochs = 5
    batch_size = 10
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

    handler.train()
    handler.test()

