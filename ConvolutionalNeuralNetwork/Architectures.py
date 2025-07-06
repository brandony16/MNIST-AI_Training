from ConvolutionalNeuralNetwork.Layers.Conv2DLayer import Conv2D
from ConvolutionalNeuralNetwork.Layers.BatchNormLayer import BatchNorm2D
from ConvolutionalNeuralNetwork.Layers.ActivationLayers import ReLU
from ConvolutionalNeuralNetwork.Layers.DenseLayer import Dense
from ConvolutionalNeuralNetwork.Layers.DropoutLayer import Dropout
from ConvolutionalNeuralNetwork.Layers.FlattenLayer import Flatten
from ConvolutionalNeuralNetwork.Layers.PoolingLayer import MaxPool2D
from ConvolutionalNeuralNetwork.Layers.SoftmaxCELayer import SoftmaxCrossEntropy

MNIST_PARAMETERS = [
    # Conv block 1
    lambda: Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
    lambda: BatchNorm2D(num_features=6),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),
    # Conv block 2
    lambda: Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
    lambda: BatchNorm2D(num_features=16),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),
    # Flatten
    Flatten,
    # FC block 1
    lambda: Dense(16 * 4 * 4, 120),
    ReLU,
    lambda: Dropout(0.2),
    # FC block 2
    lambda: Dense(120, 84),
    ReLU,
    lambda: Dropout(0.2),
    # Classifier
    lambda: Dense(84, 10),
    SoftmaxCrossEntropy,
]

CIFAR_PARAMETERS = [
    # # Conv block 1
    # lambda: Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    # # lambda: BatchNorm2D(16),
    # ReLU,
    # lambda: MaxPool2D(pool_size=2, stride=2),
    # # Conv block 2
    # lambda: Conv2D(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
    # # lambda: BatchNorm2D(32),
    # ReLU,
    # lambda: MaxPool2D(pool_size=2, stride=2),
    # # Flatten and FCs
    # Flatten,
    # lambda: Dense(64 * 8 * 8, 512),
    # ReLU,
    # # lambda: Dropout(0.5),
    # lambda: Dense(512, 256),
    # ReLU,
    # # lambda: Dropout(0.5),
    # lambda: Dense(256, 128),
    # ReLU,
    # # lambda: Dropout(0.5),
    # # Final classification layer
    # lambda: Dense(128, 10),
    # SoftmaxCrossEntropy,
    # Conv block 1
    lambda: Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    ReLU,
    lambda: MaxPool2D(2, 2),
    lambda: Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    ReLU,
    lambda: MaxPool2D(2, 2),
    Flatten,
    lambda: Dense(64 * 8 * 8, 128),
    ReLU,
    lambda: Dense(128, 10),
    SoftmaxCrossEntropy,
]
