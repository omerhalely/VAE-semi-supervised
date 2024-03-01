# Train and test VAE model for 100 MNIST labeled images
python3 main.py --train-size 100 --data "MNIST" --epochs 3 --batch-size 10 --train-mode True
python3 main.py --train-size 100 --data "MNIST" --batch-size 10

# Train and test VAE model for 600 MNIST labeled images
python3 main.py --train-size 600 --data "MNIST" --epochs 3 --batch-size 10 --train-mode True
python3 main.py --train-size 600 --data "MNIST" --batch-size 10

## Train and test VAE model for 1000 MNIST labeled images
#python3 main.py --train-size 1000 --data "MNIST" --epochs 30 --batch-size 10 --train True
#python3 main.py --train-size 1000 --data "MNIST" --batch-size 10 --train False
#
## Train and test VAE model for 3000 MNIST labeled images
#python3 main.py --train-size 3000 --data "MNIST" --epochs 30 --batch-size 10 --train True
#python3 main.py --train-size 3000 --data "MNIST" --batch-size 10 --train False
#
## Train and test VAE model for 100 FashionMNIST labeled images
#python3 main.py --train-size 100 --data "FashionMNIST" --epochs 30 --batch-size 10 --train True
#python3 main.py --train-size 100 --data "FashionMNIST" --batch-size 10 --train False
#
## Train and test VAE model for 600 FashionMnist labeled images
#python3 main.py --train-size 600 --data "FashionMNIST" --epochs 30 --batch-size 10 --train True
#python3 main.py --train-size 600 --data "FashionMNIST" --batch-size 10 --train False
#
## Train and test VAE model for 1000 FashionMnist labeled images
#python3 main.py --train-size 1000 --data "FashionMNIST" --epochs 30 --batch-size 10 --train True
#python3 main.py --train-size 1000 --data "FashionMNIST" --batch-size 10 --train False
#
## Train and test VAE model for 3000 FashionMnist labeled images
#python3 main.py --train-size 3000 --data "FashionMNIST" --epochs 30 --batch-size 10 --train True
#python3 main.py --train-size 3000 --data "FashionMNIST" --batch-size 10 --train False


