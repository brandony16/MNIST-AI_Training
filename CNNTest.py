from loadMNIST import load_and_preprocess_mnist
import numpy as np
from ConvolutionalNeuralNetwork.CNNClass import CNN
from ConvolutionalNeuralNetwork.ConvolutionalLayer import ConvLayer
from ConvolutionalNeuralNetwork.DenseLayer import Dense
from ConvolutionalNeuralNetwork.PoolingLayer import Pooling
from ConvolutionalNeuralNetwork.ReLUClass import ReLU
from ConvolutionalNeuralNetwork.Flatten import flatten

def calculateAccuracy(predictions, labels):
    labels_argmax = np.argmax(labels, axis=1)
    correct = np.sum(predictions == labels_argmax)
    total = len(labels)
    accuracy = correct / total
    return accuracy

trainImages, trainLabels, testImages, testLabels = load_and_preprocess_mnist()
trainImages = trainImages.reshape(-1, 28, 28)
testImages = testImages.reshape(-1, 28, 28)
print('Data loaded')

architecture = [
    ConvLayer(num_filters=6, filter_size=5),
    ReLU(),
    # Pooling(),
    # ConvLayer(numFilters=16, filterSize=5),
    # ReLU(),
    # Pooling(),
    # ConvLayer(numFilters=120, filterSize=5),
    # ReLU(),
    flatten(),
    Dense(input_size=576, output_size=15, activation='relu'),
    Dense(input_size=15, output_size=10, activation='softmax')
]

cnn = CNN(layers=architecture)

print("Training Started")
cnn.train(trainImages, trainLabels, epochs=3, learn_rate=0.005, batch_size=32)

predictedLabels = np.array([cnn.predict(image) for image in testImages])

accuracy = calculateAccuracy(predictedLabels, testLabels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
