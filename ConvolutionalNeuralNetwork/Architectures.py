from ConvolutionalNeuralNetwork.Layers.Conv2DLayer import Conv2D
from ConvolutionalNeuralNetwork.Layers.BatchNormLayer import BatchNorm2D
from ConvolutionalNeuralNetwork.Layers.ActivationLayers import ReLU, LeakyReLU
from ConvolutionalNeuralNetwork.Layers.DenseLayer import Dense
from ConvolutionalNeuralNetwork.Layers.DropoutLayer import Dropout
from ConvolutionalNeuralNetwork.Layers.FlattenLayer import Flatten
from ConvolutionalNeuralNetwork.Layers.PoolingLayer import MaxPool2D, AvgPool2D
from ConvolutionalNeuralNetwork.Layers.SoftmaxCELayer import SoftmaxCrossEntropy

# Layer architecture for the MNIST dataset.
MNIST_PARAMETERS = [
    # # --- Block 1: 3×3 conv ×2, 32 filters ---
    lambda: Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    ReLU,
    lambda: Conv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # 32×14×14
    # # # --- Block 2: 3×3 conv ×2, 64 filters ---
    lambda: Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    ReLU,
    lambda: Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # 64×7×7
    # # # --- Block 3: 3×3 conv ×2, 128 filters ---
    lambda: Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    ReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # 128×3×3
    lambda: AvgPool2D(pool_size=3, stride=3),  # 128×1×1
    Flatten,
    # FC block 1
    lambda: Dense(128, 256),
    ReLU,
    lambda: Dropout(0.3),
    lambda: Dense(256, 128),
    ReLU,
    lambda: Dropout(0.3),
    lambda: Dense(128, 10),
    SoftmaxCrossEntropy,
]

# Layer architecture for training a CNN on the CIFAR-10 dataset.
CIFAR_PARAMETERS = [
    # # --- Block 1: 3×3 conv ×2, 32 filters ---
    lambda: Conv2D(3, 32, kernel_size=3, padding=1),
    lambda: BatchNorm2D(32),
    LeakyReLU,
    lambda: Conv2D(32, 32, kernel_size=3, padding=1),
    lambda: BatchNorm2D(32),
    LeakyReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # 32×16×16
    # # # --- Block 2: 3×3 conv ×2, 64 filters ---
    lambda: Conv2D(32, 64, kernel_size=3, padding=1),
    lambda: BatchNorm2D(64),
    LeakyReLU,
    lambda: Conv2D(64, 64, kernel_size=3, padding=1),
    lambda: BatchNorm2D(64),
    LeakyReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # 64×8×8
    # # # --- Block 3: 3×3 conv ×2, 128 filters ---
    lambda: Conv2D(64, 128, kernel_size=3, padding=1),
    lambda: BatchNorm2D(128),
    LeakyReLU,
    lambda: Conv2D(128, 128, kernel_size=3, padding=1),
    lambda: BatchNorm2D(128),
    LeakyReLU,
    lambda: MaxPool2D(pool_size=2, stride=2),  # 128×4×4
    lambda: AvgPool2D(pool_size=4, stride=4),  # 128×1×1
    Flatten,  # 128
    # --- Classifier head ---
    lambda: Dense(128, 256),
    LeakyReLU,
    lambda: Dropout(0.2),
    lambda: Dense(256, 128),
    LeakyReLU,
    lambda: Dropout(0.2),
    lambda: Dense(128, 10),
    SoftmaxCrossEntropy,
]
