from ConvolutionalNeuralNetwork.Layers.Conv2DLayer import Conv2D
from ConvolutionalNeuralNetwork.Layers.BatchNormLayer import BatchNorm2D
from ConvolutionalNeuralNetwork.Layers.ActivationLayers import ReLU
from ConvolutionalNeuralNetwork.Layers.DenseLayer import Dense
from ConvolutionalNeuralNetwork.Layers.DropoutLayer import Dropout
from ConvolutionalNeuralNetwork.Layers.FlattenLayer import Flatten
from ConvolutionalNeuralNetwork.Layers.PoolingLayer import MaxPool2D
from ConvolutionalNeuralNetwork.Layers.SoftmaxCELayer import SoftmaxCrossEntropy

MNIST_PARAMETERS = [
    # Conv block 1: 1→32, two 3×3 convs
    lambda: Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    ReLU,
    lambda: Conv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # → 32×14×14
    # Conv block 2: 32→64, two 3×3 convs
    lambda: Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    ReLU,
    lambda: Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # → 64×7×7
    # Conv block 3: 64→128
    lambda: Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # → 128×3×3
    Flatten,
    # FC block 1
    lambda: Dense(128 * 3 * 3, 128),
    ReLU,
    lambda: Dropout(0.2),
    # FC block 2
    lambda: Dense(128, 64),
    ReLU,
    lambda: Dropout(0.2),
    # Classifier
    lambda: Dense(64, 10),
    SoftmaxCrossEntropy,
]

CIFAR_PARAMETERS = [
    # Conv block 1
    lambda: Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    lambda: BatchNorm2D(32),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),
    # Conv block 2
    lambda: Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    lambda: BatchNorm2D(64),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),
    # Flatten and FCs
    Flatten,
    lambda: Dense(64 * 8 * 8, 1024),
    ReLU,
    lambda: Dropout(0.2),
    lambda: Dense(1024, 512),
    ReLU,
    lambda: Dropout(0.2),
    # Final classification layer
    lambda: Dense(512, 10),
    SoftmaxCrossEntropy,
]
