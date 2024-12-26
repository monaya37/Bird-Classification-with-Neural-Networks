# Bird Classification with Neural Networks

This project includes two tasks that use neural networks to classify bird categories. The dataset consists of 150 rows and the following columns: **gender**, **body_mass**, **beak_length**, **beak_depth**, **fin_length**, and **bird category**. There are three bird categories: **A**, **B**, and **C**.

## Task 1: Binary Classification

In this task, the goal is to classify two bird categories based on the user's selection. The user can choose which features to use for training the model through a graphical user interface (GUI). The user can also select the algorithm type: either **Adaline** or **Perceptron**. 

For **Adaline**, the user can specify:
- **Learning rate**
- **Number of epochs**
- **MSE threshold** (used to stop the training when the error is below a certain threshold)

Both **Adaline** and **Perceptron** algorithms have been implemented from scratch. The model's performance is evaluated using a **confusion matrix**, which was also built from scratch. Additionally, the user can choose whether to **include the bias term** in the model or not.

## Task 2: Multi-Class Classification

In this task, the same dataset is used, but all features are employed to train the model, and all three bird categories (A, B, and C) are considered. The user can adjust the following parameters:
- Number of hidden neurons
- Number of hidden layers
- Number of epochs
- Learning rate
- Activation function (either **Sigmoid** or **Tanh**)

The model used for classification is a **Multi-Layer Perceptron (MLP)**, which has been built from scratch. Similarly to Task 1, the user can choose whether to **include the bias term** in the model or not.

## Data Preprocessing

Before training the models, the following preprocessing steps are applied to the data in both tasks:
- Encoding categorical columns
- Handling outliers
- Scaling numeric data
- Filling missing values (NA)

## Performance

Both tasks achieved an accuracy ranging from **95% to 100%**.

<span style="display: inline-block; text-align: center; margin-right: 10px;">
  <img src="https://github.com/monaya37/NN-Tasks/blob/b238ee6f97faf0012108eace718cbe21dba426b1/GUI.png" alt="GUI" width="500" height="400"/>
</span>

<span style="display: inline-block; text-align: center;">
  <img src="https://github.com/monaya37/NN-Tasks/blob/9265e4a9299d49fe5b039946371559496adad203/Task2%20GUI.png" alt="GUI" width="500" height="400"/>
</span>

## Thank You

This project was built with my team ([@SalmaNasrEldin](https://github.com/SalmaNasrEldin), [@marwa-ehab](https://github.com/marwa-ehab), [@SmaherNabil](https://github.com/SmaherNabil), [@zeinabsakran77](https://github.com/zeinabsakran77), [@sa10ma](https://github.com/sa10ma)). We also worked on another neural network project together—click [here](link) to see our really neat work! 🤓
