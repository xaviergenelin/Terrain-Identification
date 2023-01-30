# Terrain-Identification

For a more in depth look at the project you can look at the [report](https://github.com/xaviergenelin/Terrain-Identification/blob/main/CompetitionProject_Report.pdf) that we wrote.

## Purpose

This project examines accelerometer and gyroscope readings from a prosthetic leg to predict the surface a subject is walking on. The goal is to build the best model to predict the terrain of the limb for a competition on a hidden dataset. This gave us a classification problem with four different terrain types: solid surface (class 0), descending stairs (class 1), ascending stairs (class 2), and grass (class 3). 

## Dependencies
* pandas
* pickle
* torch
* numpy

## Dataset Description

This was a course project where data was provided for us to build and train the model, data to test our model, and a hidden dataset used for the competition portion. The dataset has X, Y, and Z datapoints for the acceleromoeter and gyroscope measurements from the limb. The data is also imbalanced for the different terrains. 

## Deep Learning Method

For this task, we used a Convulutional Neural Network that has two covolutional layers and a fully connected linear layer. The full architecture is shown here
![](https://github.com/xaviergenelin/Terrain-Identification/blob/main/images/Architecture.jpg)

The first convolutional layer has 6 input channels for the X, Y, and Z values in the two sensors. The outputs are fed through a ReLU activation function. The activations are pooled using a kernal size of 3 and stride of 1 for the first layer and then a kernal size of 3 and stride of 3 for the second layer. The second layer's output is also fed to a ReLU activation function. The loss function is calculated using Cross Entropy Loss. The weights are updated after each iteration using stochastic gradient descent, with a learning rate of 0.1 and momentum factor of 0.9.

### Model Selection

For our hyperparemeters, we used 5-fold cross validation. We considered batch sizes of 32 or 64, kernal sizes of 5, 7, or 9, and output sizes of 6 or 12 for each convolutional layer. This gives us a total of 72 models to evaluate. These were evaluated using accuracy and macro-averaged F1 score that gave us the following results:
<br>
Accuracy
![accuracy](https://github.com/xaviergenelin/Terrain-Identification/blob/main/images/ModelAccuracy.png)
Macro F1
![macro F1](https://github.com/xaviergenelin/Terrain-Identification/blob/main/images/ModelMacroF1.png)

Both measures had the 71st model performing the best, which had the following hyperparameters:
| Hyperparameter    | Value |
|-------------------|-------|
| Batch Size        | 64    |
| Conv1 Kernel Size | 9     |
| Conv1 N Filters   | 12    |
| Conv2 Kernel Size | 9     |
| Conv2 N Filters   | 6     |

To determine the number of epochs to use, we looked at the learning curve for each of the 5 folds at all 16 epochs, with the learning curve below:
![](https://github.com/xaviergenelin/Terrain-Identification/blob/main/images/LearningCurve.png)

## Results

After selecting the best model, we retrained the model for 5 epochs using the full training set. Evaluating the model with the held-out test data, gave us the following results:
<br>
![](https://github.com/xaviergenelin/Terrain-Identification/blob/main/images/TestResults.png)

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.91      | 0.94   | 0.92     | 48949   |
| 1            | 0.48      | 0.90   | 0.63     | 1591    |
| 2            | 0.90      | 0.92   | 0.91     | 2991    |
| 3            | 0.85      | 0.68   | 0.76     | 13552   |
| Accuracy     |           |        | 0.88     | 67083   |
| Macro Avg    | 0.79      | 0.86   | 0.80     | 67083   |
| Weighted Avg | 0.89      | 0.88   | 0.88     | 67083   |

We achieved an accuracy of 88% and a macro-averaged F1 of 0.8. We had a high error rate for class 1 (ascending stairs) and class 3 (walking on grass). The low precision and high recall means we overpredicted class 1. Class 3, on the other hand, has a high precision and low recall, indicating that it was under-predicted.
