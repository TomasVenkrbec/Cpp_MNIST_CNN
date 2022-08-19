# Convolutional Neural Network for MNIST hand-written digit classification with pure C++
An implementation of deep learning framework in C++, without the use of any external libraries, inspired by [Keras](https://keras.io/). The framework is then used to create convolutional neural network, which is used to classify hand-written numbers from [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

## Features
As already mentioned, the project structure is inspired by Keras. Created framework is very modular, and all types of networks can be built very easily. The project is not limited to MNIST dataset only, and new datasets can be added in similar way, without interfering with other parts of project. There exists number of already implemented activations, callbacks, neuron initializers, layers, losses, optimizers and regularizers, allowing for multi-purpose network creation. Adding any new modules is also simple.

## MNIST dataset
It's necessary to download MNIST dataset in CSV form in order for project to work. The dataset needs to be put in the `mnist_csv/` folder and consist of two files, `mnist_train.csv` and `mnist_test.csv`. The most reliable way to make this work is to run the provided script, `get_dataset.sh`.

## Current state of project
- Fully implemented dataset loading, network creation and inference for fully-connected and convolutional networks
- Backpropagation only for fully-connected networks

## Planned features
- Backpropagation for convolutional layers

## Achieved results
Fully-connected network on MNIST - **87 % validation accuracy** 