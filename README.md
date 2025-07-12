### MNIST AI TRAINING

A project to familiarize myself with different forms of machine learning and AI, all trained on the MNIST and CIFAR-10 data sets.
Also practiced using data visualization libraries like matplotlib to gain a deeper understanding of the models and how they progressed over time.

## Table of Contents
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Data Preparation](#data-preparation)
- [Data Visualization](#data-visualization)
- [Forms of AI](#forms-of-ai)
- [K-Nearest Neighbors](#k-nearest-neighbors-knn)
- [Random Forest](#random-forest)
- [Multilayer Perceptron / Neural Network](#multilayer-perceptron--neural-network)
- [Convolutional Neural Network](#convolutional-neural-network-cnn)

### Getting Started

To clone this repository, run the following command in your terminal:

```bash
git clone https://github.com/brandony16/MNIST-AI_Training
```

Install dependencies:
```bash
pip install -r requirements.txt
```

(Optional) Install in editable mode:
```bash
pip install -e .
``` 

## Datasets
For this project, I trained my models on the MNIST and CIFAR-10 datasets. MNIST is a simple dataset that most classification models can perform well on. As such, all models achieved an accuracy over 96% on MNIST. Due to the accuracies being very high and the dataset not being very challenging, it can be difficult to compare models. 
For this reason, I decided to also train the models on the CIFAR-10 dataset. This is a much more complex dataset that can challenge the models to a greater extent, showcasing their differences and strengths. 

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
- K-Nearest Neigbors (KNN)
- Random Forest
- Multilayer Perceptron
- Convolutional Neural Network

## K-Nearest Neighbors (KNN)
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

### Optimizations
All optimizations to KNN were to speed up the algorithm, as there is little you can do to improve accuracy. To speed the algorithm up, I used numba NJIT. This allowed the whole function to run for me in ~10 seconds on MNIST and CIFAR.  

### Results
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

### Optimizations
Once again, most optimizations were made for run time instead of accuracy. The most taxing function is the best_split function that determines the best split for a given tree. For this reason, I used NJIT to optimize it.

### Results
Using a Random Forest classifier, I was able to achieve a accuracy of 96.72% on MNIST and 46.78% on CIFAR-10.
![Confusion Matrix and Precision/Recall/F1 score bar chart for RF model trained on MNIST](./screenshots/MNIST/RF100(maxd25,maxf28,msl1)(96.72).png)
*Confusion matrix and Precision/Recall/F1 score bar chart after training on MNIST trees=100, max-depth=25, max-feat=28, min-samp-leaf=1.*

![Confusion Matrix and Precision/Recall/F1 score bar chart for KNN on CIFAR](./screenshots/CIFAR/RF300(maxd=25maxf=64)(46.78).png)
*Confusion matrix and Precision/Recall/F1 score bar chart after training on CIFAR with with trees=300, max-depth=25, max-feat=64, min-samp-leaf=2.*

## Multilayer Perceptron / Neural Network
This model uses layers of connected nodes to classify input data. Unlike previous models, this one "learns" over many passes of the training data, called epochs. Click [Here](https://en.wikipedia.org/wiki/Multilayer_perceptron) to learn more.

To run the Neural Network script:
```bash
python ./NeuralNetwork/NNMain.py
```
This program has arguments for --dataset, --valid-split, --batch-size, --epochs, --lr, --lr-drop-every, --lr-drop-factor, --sample-size, --opt, and --activ.
More info can be found by running:
```bash
python ./NeuralNetwork/NNMain.py --help
```
or looking at the NNMain file.

### Optimizations
As the first model that learns over epochs, there are many more optimizations that can be made to improve accuracy. 
Some include:
- **Batch Norm:** This layer normalizes each feature to have zero mean and unit variance during training. This helps stabilize and speed up training.
- **Dropout:** This layer zeros out a percentage of data between each dense layer. This forces the model to not rely too much on single nodes, reducing overfitting and improving accuracy. 
- **Optimization Functions:** There are 2 optimization functions available, Adam and SGD. Adam often learns faster convergence with less tuning, while SGD often leads to better final results. Common learning rates are 0.01 or 0.001 for Adam and 0.1 or 0.01 for SGD

### Results
Using a Neural Network, I was able to achieve a accuracy of 99.00% on MNIST and 45.25% on CIFAR-10.
![Charts for Neural Network trained on MNIST](./screenshots/MNIST/NN(99.00)(1024,%20512,%20256,%20128)(0.001lr%20adam%200.2drop).png)
*Performance Charts after training on MNIST with layer sizes [2048, 1024, 512, 256, 128], lr=0.001, opt=adam, batch-size=64*

![Charts for Neural Network trained on CIFAR](./screenshots/CIFAR/NN(45.25)(layers=[2048,1024,1024,512,128],epochs=50,batch=64,lr=0.1,opt=sgd,activ=leakyRelu).png)
*Performance Charts after training on CIFAR for 50 epochs with layer sizes [2048, 1024, 512, 256, 128], lr=0.1, opt=sgd, batch-size=64*

## Convolutional Neural Network (CNN)
This model is similar to the basic Neural Network, but it first includes convolutional layers. These layers learn kernels that can pick out relevant features from input data. This makes it much more powerful for image classification as it can find structures in images instead of looking at individual pixels. Click [Here](https://en.wikipedia.org/wiki/Convolutional_neural_network) to learn more.

To run the Neural Network script:
```bash
python ./ConvolutionalNeuralNetwork/CNNMain.py
```
This program has arguments for --dataset, --valid-split, --batch-size, --epochs, --lr, --sample-size, and --opt.
More info can be found by running:
```bash
python ./ConvolutionalNeuralNetwork/CNNMain.py --help
```
or looking at the CNNMain file.

### Optimizations
The CNN uses many of the optimizations from the [Neural Network](#optimizations-2) section, with a few more added. Batch Norm serves the same purpose but is now done in 2D instead of 1D. 
- **Pooling:** This layer reduces the dimensionality after convolution blocks. This helps reduce model size and make the model less sensitive to noise in the data.
- **LR Scheduler:** This is an function that adapts the learning rate over time for maximum performance. It an implementation of [OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html), which build the learning rate up in a warm up phase, then tapers it off over time using cosine annealing. 
- **Weight Clipping:** When training on CIFAR, I was gettting very unstable results, with loss jumping up and down. Because of this, I implemented weight clipping, which prevents weights from blowing up and helps stabilize training.

### Results
Using a CNN, I was able to achieve a accuracy of 99.36% on MNIST and 56.94% on CIFAR-10. The result on MNIST is good, but the result on CIFAR is lower than expected. I found training on CIFAR to be very unstable, and despite tweaking and testing with many architectures, I could not get a high accuracy. A CNN should be able to reach 70-80% fairly easily. Regardless, the result of 56.94% is still 10% higher than any other model was able to achieve, showing the CNN's strengths.
![Charts for CNN trained on MNIST](./screenshots/MNIST/CNN(99.36)(epochs=20,lr=0.1,batch=64,opt=sgd).png)
*Performance Charts after training on MNIST with lr=0.1, opt=sgd, epochs=20, batch-size=64.*

![Charts for Neural Network trained on CIFAR](./screenshots/CIFAR/CNN(56.94).png)
*Performance Charts after training on CIFAR for 50 epochs with lr=0.1, opt=sgd, batch-size=256, epochs=50*