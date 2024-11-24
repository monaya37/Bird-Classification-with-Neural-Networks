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
# task2 functions
class Task2Functions:

    def __init__(self, gui):
        self.gui = gui

        self.epochs = None
        self.learning_rate = None
        self.include_bias = None
        self.activation_function = None
        self.dataset = None

        self.min = None
        self.max = None
        self.weights = None
        self.bias = None


    # Functions
    def run_algorithm(self):

        #initilize global variables
        self.dataset = read_file('birds.csv')
        self.epochs = int(self.gui.get_epochs())
        self.learning_rate = float(self.gui.get_learning_rate())
        self.include_bias = self.gui.get_bias_state()
        self.activation_function = self.gui.get_algorithm_type()
        self.gui.show_selected()

        if(self.activation_function == 'Sigmoid'):
            return
        else:
            return


    def train_model(self, X_train, y_train):
        #TO-DO
        return
            
    def sigmoid(self, x):
        return 

    def tanh(self, x):
        return 
   
    def update_weights_and_bias(self, X, error):

        if self.include_bias:
            self.weights = self.weights + self.learning_rate * error * X
            self.bias = self.bias + self.learning_rate * error
        else:
            self.weights = self.weights + self.learning_rate * error * X

        return self.weights, self.bias

    def compute_error(self, X, y):

        y_predict = np.dot(self.weights, X) + self.bias
        y_predict = self.activation_function(y_predict)
        error = y - y_predict

        return error

    #predict test
    def predict(self, X):

        predictions = []
        X = X.values

        for i in range(X.shape[0]):
            y_predict = np.dot(self.weights, X[i]) + self.bias
            y_predict = self.activation_function(y_predict)
            predictions.append(y_predict)

        predictions = np.array(predictions)


        return predictions


    def split_to_train_test(self, dataset):
        #TO-DO
        train_data = []
        test_data = []

        #return X_train, X_test, y_train, y_test
        return


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