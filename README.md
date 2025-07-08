### MNIST AI TRAINING

A project to familiarize myself with different forms of machine learning and AI, all trained on the MNIST and CNN data sets.
Also practiced using data visualization libraries like matplotlib to gain a deeper understanding of the models. 

## Datasets
For this project, I trained my models on the MNIST and CIFAR-10 datasets.
- **MNIST**
    - Dataset of 28x28 images of hand-drawn numbers 0-9.
    - All images are grayscale.
    - 70,000 total images
- **CIFAR-10**
    - Dataset of 32x32 images of objects and animals.
    - 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
    - All color images.
    - 60,000 total images

## Data Preparation


## Data Visualization


## Forms of AI
- K Nearest Neigbors (KNN)
- Random Forest
- Multilayer Perceptron
- Convolutional Neural Network

## KNN
- This model plots training data in space.
- When predicting on test data, it finds the closest K "neighbors" to it and 
performs a majority vote among those points to determine what to classify the new point as.

To run the KNN script:
```bash
python KNN/KNNMain.py {K}
```
Where K is the number of neighbors (default: 3)

Using KNN, I was able to achieve a accuracy of 97.72% on MNIST and XX.XX% on CIFAR-10
IMAGE OF CONFUSION MATRIX AND BAR CHARTS

