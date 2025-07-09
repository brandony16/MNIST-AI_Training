### MNIST AI TRAINING

A project to familiarize myself with different forms of machine learning and AI, all trained on the MNIST and CIFAR-10 data sets.
Also practiced using data visualization libraries like matplotlib to gain a deeper understanding of the models and how they progressed over time.

## Datasets
For this project, I trained my models on the MNIST and CIFAR-10 datasets.
- **MNIST**
    - Dataset of 28x28 images of hand-drawn numbers 0-9.
    - All images are grayscale.
    - 70,000 total images
- **CIFAR-10**
    - Dataset of 32x32 images of objects and animals.
    - 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck
    - All color images.
    - 60,000 total images

## Data Preparation
Datasets were loaded using sklearn's fetch_openml function, then cached locally. 
To prepare the data for the models, multiple steps were taken:
1. Data was normalized to the [0, 1] range
2. The labels were one-hot encoded if necessary
3. Data was split using train_test_split from sklearn
4. Data was converted to correct types and reshaped into the correct size
5. Data was normalized for the dataset

## Data Visualization
After training, the models performance is visualized using matplotlib. Four graphs are created: Loss vs Epoch, Accuracy vs Epoch, a confusion matrix, and a precision/recall/f1-score bar graph. 
- The loss and accuracy graphs are created with stored metrics during training and are not applicable to KNN or RandfomForest as those do not train over epochs. The loss is calcuated using the cross-entropy loss formula.
- The confusion matrix shows what the model predicted vs what the image actually was for each class. This helps pinpoint which classes the model is confusing other classes for and gives helpful per-class statistics.
- The bar chart shows the precision, recall, and f1-score for each class.
  - Precision is the rate of how many predicted positives are actually correct. It is calculated using the formula: TP / (TP + FP).
  - Recall is the rate of how many actual positives were correctly predicted. It is calculated using the formula: TP / (TP + FN).
  - F1 Score is the harmonic mean of precision and recall and is calculated with the formula: 2 * (Precision * Recall) / (Precision + Recall)

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

