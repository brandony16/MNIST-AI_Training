from loadMNIST import load_and_preprocess_mnist
import numpy as np
from ConvolutionalNeuralNetwork.CNNClass import CNN
from ConvolutionalNeuralNetwork.ConvolutionalLayer import ConvLayer
from ConvolutionalNeuralNetwork.DenseLayer import Dense
from ConvolutionalNeuralNetwork.PoolingLayer import Pooling
from ConvolutionalNeuralNetwork.ReLUClass import ReLU

def calculateAccuracy(predictions, labels):
    correct = np.sum(predictions == labels)
    total = len(labels)
    accuracy = correct / total
    return accuracy

trainImages, trainLabels, testImages, testLabels = load_and_preprocess_mnist()
trainImages = trainImages.reshape(-1, 28, 28)
testImages = testImages.reshape(-1, 28, 28)
print('Data loaded')

architecture = [
    ConvLayer(numFilters=6, filterSize=5),
    ReLU(),
    # Pooling(),
    # ConvLayer(numFilters=16, filterSize=5),
    # ReLU(),
    # Pooling(),
    # ConvLayer(numFilters=120, filterSize=5),
    # ReLU(),
    Dense(inputSize=6, outputSize=15, activation='relu'),
    Dense(inputSize=15, outputSize=10, activation='softmax')
]

cnn = CNN(layers=architecture)

print("Training Started")
cnn.train(trainImages, trainLabels, epochs=3, learn_rate=0.005, batch_size=32)

predictedLabels = np.array([cnn.predict(image) for image in testImages])

accuracy = calculateAccuracy(predictedLabels, testLabels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
