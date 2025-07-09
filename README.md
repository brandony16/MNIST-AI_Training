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
This model plots training data in space. When predicting on test data, it finds the closest K "neighbors" to it and performs a majority vote among those points to determine what to classify the new point as. Click [here](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) for more info.

To run the KNN script:
```bash
python KNN/KNNMain.py
```
This program has arguments for --dataset, --valid-split, --k, and --num-dims.
More info can be found by running:
```bash
python KNN/KNNMain.py --help
```
or looking at the KNNMain file.

Using a KNN classifier, I was able to achieve a accuracy of 97.72% on MNIST and 39.39% on CIFAR-10.
![Confusion Matrix and Precision/Recall/F1 score bar chart for KNN model trained on MNIST](./screenshots/MNIST/KNN3(97.72).png)
*Confusion matrix and Precision/Recall/F1 score bar chart after training on MNIST with k=3 and num_dims=50.*

![Confusion Matrix and Precision/Recall/F1 score bar chart for KNN on CIFAR](./screenshots/CIFAR/KNN13(39.39).png)
*Confusion matrix and Precision/Recall/F1 score bar chart after training on CIFAR with k=13 and num_dims=50.*

## Random Forest
This model creates many decision trees, each trained on different subsets of the data and features. By combining the outputs of many diverse trees, accuracy is improved and overfitting is reduced. Click [Here](https://en.wikipedia.org/wiki/Random_forest) to learn more.

To run the RandomForest script:
```bash
python ./RandomForest/RFMain.py
```
This program has arguments for --dataset, --valid-split, --trees, --max-depth, --max-feat, --njobs, and --min-samp-leaf.
More info can be found by running:
```bash
python ./RandomForest/RFMain.py --help
```
or looking at the RFMain file.

Using a Random Forest classifier, I was able to achieve a accuracy of 96.72% on MNIST and 46.78% on CIFAR-10.
![Confusion Matrix and Precision/Recall/F1 score bar chart for RF model trained on MNIST](./screenshots/MNIST/RF100(maxd25,maxf28,msl1)(96.72).png)
*Confusion matrix and Precision/Recall/F1 score bar chart after training on MNIST trees=100, max-depth=25, max-feat=28, min-samp-leaf=1.*

![Confusion Matrix and Precision/Recall/F1 score bar chart for KNN on CIFAR](./screenshots/CIFAR/RF300(maxd=25maxf=64)(46.78).png)
*Confusion matrix and Precision/Recall/F1 score bar chart after training on CIFAR with with trees=300, max-depth=25, max-feat=64, min-samp-leaf=2.*

## Neural Network
This model uses layers of connected nodes to classify input data. Unlike previous models, this one "learns" over many passes of the training data, called epochs. Click [Here](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) to learn more.

To run the Neural Network script:
```bash
python ./NeuralNetwork/NNMain.py
```
This program has arguments for --dataset, --valid-split, --batch-size, --epochs, --lr, --lr-drop-every, --lr-drop-factor, --sample-size, --opt, and --activ.
More info can be found by running:
```bash
python ./NeuralNetwork/NNMain.py --help
```
or looking at the RFMain file.

Using a Neural Network, I was able to achieve a accuracy of 99.00% on MNIST and 46.78% on CIFAR-10.
![Charts for Neural Network trained on MNIST](./screenshots/MNIST/NN(99.00)(1024,%20512,%20256,%20128)(0.001lr%20adam%200.2drop).png)
*Performance Charts after training on MNIST with layer sizes [2048, 1024, 512, 256, 128], lr=0.001, opt=adam.*

![Charts for Neural Network trained on CIFAR](./screenshots/CIFAR/RF300(maxd=25maxf=64)(46.78).png)
*Performance Charts after training on MNIST with layer sizes [2048, 1024, 512, 256, 128], lr=0.001, opt=adam*