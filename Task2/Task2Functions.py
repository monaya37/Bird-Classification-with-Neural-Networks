import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder    
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from random import shuffle
from sklearn.utils import shuffle
import seaborn as sns
from Functions import *
import math


# task2 functions
class Task2Functions:

    def __init__(self, gui):
        self.gui = gui

        self.epochs = None
        self.learning_rate = None
        self.include_bias = None
        self.activation_function = None
        self.dataset = None
        self.classes = None
        self.features = None

        self.min = None
        self.max = None
        self.weights = None
        self.bias = None


    # Functions
    def run_algorithm(self):

        #initilize global variables
        self.dataset = read_file('birds.csv')
        self.features = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]
        self.classes = ["A", "B", "C"]
        self.epochs = int(self.gui.get_epochs())
        self.learning_rate = float(self.gui.get_learning_rate())
        self.include_bias = self.gui.get_bias_state()
        self.activation_function = self.gui.get_algorithm_type()

        if(self.activation_function == 'Sigmoid'):
            self.activation_function = self.sigmoid
        else:
            self.activation_function = self.tanh

        self.gui.show_selected()

        X_train, X_test, y_train, y_test = split_to_train_test(self.dataset, self.classes, self.features)

        # Preprocess
        X_train, X_test = preprocess_features(X_train, X_test, self.features)
        y_train, y_test = preprocess_target(y_train, y_test)




    def train_model(self, X, y):
        n = X.shape[0]
        self.bias = 0
        self.weights = np.random.randn(X.shape[1])    
        #مش عارف      
        for _ in range(self.epochs):
            net = np.dot(self.weights, X) + self.bias
            y_pred = self.activation_function(net)
            error = y - y_pred
            self.update_weights_and_bias(X, error)



        
            
    def sigmoid(self, net):
        return 1 / 1 + np.exp(-net)

    def tanh(self, net):
        return (2 / 1 + np.exp(-2 * net)) - 1
   
    def update_weights_and_bias(self, X, error):

        if self.include_bias:
            self.weights = self.weights + self.learning_rate * error * X
            self.bias = self.bias + self.learning_rate * error
        else:
            self.weights = self.weights + self.learning_rate * error * X

        return self.weights, self.bias


    #predict test
    def predict(self, X):

        predictions = []
        #TO-DO
        return predictions



    def plot_function(self, X, Y):
        #TO-DO
        return


    def evaluate_predictions(self, y_true, y_pred, min, max):
        TP = sum((y_true == max) & (y_pred == max))
        TN = sum((y_true == min) & (y_pred == min))
        FP = sum((y_true == min) & (y_pred == max))
        FN = sum((y_true == max) & (y_pred == min))

        # Avoid division by zero
        if np.all(TP + TN + FP + FN == 0):
            print("Warning: No predictions made.")
            accuracy = 0
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            accuracy *= 100

        #print(f"Confusion Matrix:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

        self.update_accuracy(accuracy)
        self.plot_confusion_matrix(TP, TN, FP, FN)

    def plot_confusion_matrix(self, TP, TN, FP, FN):

        confusion_matrix = np.array([[TP, FN], [FP, TN]])

        self.gui.ax2.clear()
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap= "magma", 
                    xticklabels=["Predicted Positive", "Predicted Negative"],
                    yticklabels=["Actual Positive", "Actual Negative"], ax=self.gui.ax2, cbar=False)
        
        # Set titles and labels
        self.gui.ax2.set_title("Confusion Matrix")
        self.gui.ax2.set_xlabel("Predicted Class")
        self.gui.ax2.set_ylabel("Actual Class")

        self.gui.canvas.draw()

    def update_accuracy(self, accuracy):
        self.gui.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")