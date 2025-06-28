from ConvolutionalNeuralNetwork.Layers.Conv2DLayer import Conv2D
from ConvolutionalNeuralNetwork.Layers.BatchNormLayer import BatchNorm2D
from ConvolutionalNeuralNetwork.Layers.ActivationLayers import ReLU
from ConvolutionalNeuralNetwork.Layers.DenseLayer import Dense
from ConvolutionalNeuralNetwork.Layers.DropoutLayer import Dropout
from ConvolutionalNeuralNetwork.Layers.FlattenLayer import Flatten
from ConvolutionalNeuralNetwork.Layers.PoolingLayer import MaxPool2D
from ConvolutionalNeuralNetwork.Layers.SoftmaxCELayer import SoftmaxCrossEntropy

LENET_PARAMETERS = [
    lambda: Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),
    lambda: Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),
    Flatten,
    lambda: Dense(256, 120),
    ReLU,
    lambda: Dense(120, 84),
    ReLU,
    lambda: Dense(84, 10),
    SoftmaxCrossEntropy,
]
