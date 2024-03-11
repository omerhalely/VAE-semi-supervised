# Train VAE on MNIST datasets
python main.py --model-type "VAE" --model-name "VAE_MNIST" --data "MNIST" --epochs 10 --batch-size 10 --train-mode True

# Train Classifiers on the latent representation of VAE_MNIST model
python main.py --model-type "Classifier" --model-name "Classifier_100_MNIST" --train-size 100 --vae-model-name "VAE_MNIST" --data "MNIST" --train-mode True
python main.py --model-type "Classifier" --model-name "Classifier_600_MNIST" --train-size 600 --vae-model-name "VAE_MNIST" --data "MNIST" --train-mode True
python main.py --model-type "Classifier" --model-name "Classifier_1000_MNIST" --train-size 1000 --vae-model-name "VAE_MNIST" --data "MNIST" --train-mode True
python main.py --model-type "Classifier" --model-name "Classifier_3000_MNIST" --train-size 3000 --vae-model-name "VAE_MNIST" --data "MNIST" --train-mode True

# Train VAE on FashionMNIST datasets
python main.py --model-type "VAE" --model-name "VAE_FashionMNIST" --data "FashionMNIST" --epochs 10 --batch-size 10 --train-mode True

# Train Classifiers on the latent representation of VAE_FashionMNIST model
python main.py --model-type "Classifier" --model-name "Classifier_100_FashionMNIST" --train-size 100 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST" --train-mode True
python main.py --model-type "Classifier" --model-name "Classifier_600_FashionMNIST" --train-size 600 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST" --train-mode True
python main.py --model-type "Classifier" --model-name "Classifier_1000_FashionMNIST" --train-size 1000 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST" --train-mode True
python main.py --model-type "Classifier" --model-name "Classifier_3000_FashionMNIST" --train-size 3000 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST" --train-mode True

# Test MNIST classifiers
python main.py --model-type "Classifier" --model-name "Classifier_100_MNIST" --train-size 100 --vae-model-name "VAE_MNIST" --data "MNIST"
python main.py --model-type "Classifier" --model-name "Classifier_600_MNIST" --train-size 600 --vae-model-name "VAE_MNIST" --data "MNIST"
python main.py --model-type "Classifier" --model-name "Classifier_1000_MNIST" --train-size 1000 --vae-model-name "VAE_MNIST" --data "MNIST"
python main.py --model-type "Classifier" --model-name "Classifier_3000_MNIST" --train-size 3000 --vae-model-name "VAE_MNIST" --data "MNIST"

# Test FashionMNIST classifiers
python main.py --model-type "Classifier" --model-name "Classifier_100_FashionMNIST" --train-size 100 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST"
python main.py --model-type "Classifier" --model-name "Classifier_600_FashionMNIST" --train-size 600 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST"
python main.py --model-type "Classifier" --model-name "Classifier_1000_FashionMNIST" --train-size 1000 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST"
python main.py --model-type "Classifier" --model-name "Classifier_3000_FashionMNIST" --train-size 3000 --vae-model-name "VAE_FashionMNIST" --data "FashionMNIST"
