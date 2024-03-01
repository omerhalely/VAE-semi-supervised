import argparse
import torch
from Handler import Handler


parser = argparse.ArgumentParser(
    description="A program to build and train the VAE semi supervised model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--train-size",
    type=int,
    help="The amount of labeled training images which will be used for the training process. Default - 100.",
    default=100,
)

parser.add_argument(
    "--data",
    type=str,
    help="The amount of labeled training images which will be used for the training process (MNIST/FashionMNIST)."
         " Default - MNIST",
    default="MNIST",
)

parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs for training. Default - 30",
    default=30,
)

parser.add_argument(
    "--batch-size",
    type=int,
    help="Batch size. Default - 10",
    default=10,
)

parser.add_argument(
    "--train-mode",
    type=bool,
    help="If set to True, model will be trained, else, model will be loaded and tested. Default - False",
    default=False,
)


if __name__ == "__main__":
    args = parser.parse_args()
    train_size = args.train_size
    data_type = args.data
    model_name = f"VAE_{train_size}_{data_type}"
    height = 28
    width = 28
    hidden_size = 256
    latent_size = 10
    lr = 0.001
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train = args.train_mode

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
    if train:
        handler.train()
    else:
        handler.test()
