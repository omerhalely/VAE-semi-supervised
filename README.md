# VAE-semi-supervised
Semi-supervised learning with deep generative models. We train an Encoder-Decoder network. The encoder takes an image
as input and outputs a vector. The decoder takes the output vector of the encoder and outputs a reconstructed image.
After training the VAE network, we train a SVM classifier which takes the output vector of the encoder and outputs
a classification.

## Training
For training a VAE model:
```bash
python main.py --model-type "VAE" --model-name "model-name" --data "data-type" --epochs 10 --batch-size 10 --train-mode True
```
model-type can be VAE/Classifier. When training a VAE, we set the model type to VAE.

model-name is the name of the model which will be saved to ./models/model-name.

data is the data which will be used for training. Can be MNIST/FashionMNIST.

For example, training a VAE model over the MNIST dataset:
```bash
python main.py --model-type "VAE" --model-name "VAE_MNIST" --data "MNIST" --epochs 10 --batch-size 10 --train-mode True
```

After training the VAE, we train a SVM classifier:
```bash
python main.py --model-type "Classifier" --model-name "model-name" --train-size 100 --vae-model-name "VAE_model_name" --data "data-type" --train-mode True
```
model-type can be VAE/Classifier. When training a classifier, we set the model type to Classifier.

model-name is the name of the model which will be saved to ./models/model-name.

train-size is the number of labeled examples which will be used for training the SVM classifier.

vae-model-name is the name of the VAE model which will be used to get the latent representation of the input image.

data is the data which will be used for training. Can be MNIST/FashionMNIST.


For example, training a classifier over the MNIST dataset:
```bash
python main.py --model-type "Classifier" --model-name "Classifier_100_MNIST" --train-size 100 --vae-model-name "VAE_MNIST" --data "MNIST" --train-mode True
```

## Testing
For testing a classifier:
```bash
python main.py --model-type "Classifier" --model-name "model-name" --train-size 100 --vae-model-name "VAE-model-name" --data "data-type"
```
For example, testing a classifier over 100 labeled examples of the MNIST dataset:
```bash
python main.py --model-type "Classifier" --model-name "Classifier_100_MNIST" --train-size 100 --vae-model-name "VAE_MNIST" --data "MNIST"
```
