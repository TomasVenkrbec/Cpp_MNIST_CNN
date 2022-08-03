# Convolutional Neural Network for MNIST hand-written digit classification with pure C++
An implementation of Convolutional Neural Network in C++, without the use of any external libraries, inspired by [Keras](https://keras.io/). The network is then used to classify hand-written numbers from [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

## MNIST dataset
It's necessary to download MNIST dataset in CSV form in order for project to work. The dataset needs to be put in the `mnist_csv/` folder and consist of two files, `mnist_train.csv` and `mnist_test.csv`. The most reliable way to make this work is to run the provided script, `get_dataset.sh`.