import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
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

        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden = None
        self.bias_output = None


    def run_algorithm(self):

        #initialize global variables
        self.initialize_variables()
        
        # Split based on selected
        X_train, X_test, y_train, y_test = split_to_train_test(
            self.dataset, self.classes, self.features)

        # preprocess
        X_train, X_test = preprocess_features(X_train, X_test, self.features)
        y_train, y_test = self.preprocess_target(y_train, y_test)
        #y_train ==> [0,0,1] one hot encoded

        input_size = X_train.shape[1]
        output_size = len(self.classes)

        #train
        self.train_model(X_train, y_train, input_size, output_size)

        #test
        y_pred = self.predict_neural_network(X_test)
        # confusion metrics
        self.evaluate_predictions(y_test, y_pred)


    def initialize_variables(self):
        #initialize global variables
        self.dataset = read_file('birds.csv')
        self.epochs = int(self.gui.get_epochs())
        self.learning_rate = float(self.gui.get_learning_rate())
        self.include_bias = self.gui.get_bias_state()
        self.activation_function = self.gui.get_algorithm_type()
        if(self.activation_function == 'Sigmoid'):
            self.activation_function = self.sigmoid
            self.activation_d = self.sigmoid_derivative
        else:
            self.activation_function = self.tanh
            self.activation_d = self.tanh_derivative

        self.neurons = self.gui.get_neurons()
        self.gui.show_selected()
        self.classes = ['A', 'B', 'C']
        self.features = ['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']
        return

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def initialize_weights(self, input_size, output_size):
        hidden_size = self.num_hidden_layers

        #.uniform function generates random numbers from a uniform distribution.
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.bias_hidden = np.zeros((1, hidden_size)) # 1 bias term per neuron
        self.bias_output = np.zeros((1, output_size))


        print("self.weights_input_hidden")
        print(self.weights_input_hidden)
        print("self.weights_hidden_output")
        print(self.weights_hidden_output)
        print("self.bias_hidden")
        print(self.bias_hidden)
        print("self.bias_output")
        print(self.bias_output)

        return 

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # to avoid overflow
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward_pass(self, X):

        # hidden layer
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.activation_function(hidden_input)
        print("hidden output: ", hidden_output)


        # output layer
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output

        # apply softmax for probabilities(since it is not binary classif)
        final_output = self.softmax(final_input)
        return hidden_output, final_output

    def backward_pass(self, X, y, hidden_output, final_output):

        # output layer error and gradient
        output_error = y - final_output
        output_gradient = output_error * self.sigmoid_derivative(final_output)  # Output uses sigmoid

        # hidden layer error and GD
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.activation_d(hidden_output)


        # updates for w and b
        weights_hidden_output_update = np.dot(hidden_output.T, output_gradient)
        weights_input_hidden_update = np.dot(X.T, hidden_gradient)

        bias_output_update = np.sum(output_gradient, axis=0, keepdims=True)
        bias_hidden_update = np.sum(hidden_gradient, axis=0, keepdims=True)

        return weights_input_hidden_update, weights_hidden_output_update, bias_hidden_update, bias_output_update
    
    def train_model(self, X_train, y_train, input_size, output_size):
        
        # init weights and bias
        self.initialize_weights(input_size, output_size)

        for _ in range(self.epochs):
            # forward prop
            hidden_output, final_output = self.forward_pass(
                X_train)

            # back prop
            weights_input_hidden_update, weights_hidden_output_update, bias_hidden_update, bias_output_update = self.backward_pass(
                X_train, y_train, hidden_output, final_output)

            # update w and b after every epoch
            self.weights_input_hidden += self.learning_rate * weights_input_hidden_update
            self.weights_hidden_output += self.learning_rate * weights_hidden_output_update
            self.bias_hidden += self.learning_rate * bias_hidden_update
            self.bias_output += self.learning_rate * bias_output_update

        return

    def predict_neural_network(self, X):
        # forward prop 4 predictions
        _, final_output = self.forward_pass(X)

        # index of the class with the highest prob
        y_pred = np.argmax(final_output, axis=1)
        print("y_pred", y_pred)
        return y_pred
    
    def preprocess_target(self, y_train, y_test):

        encoder = OneHotEncoder(sparse_output=False)  # sparse=False ensures that the output is a dense array

        # Convert pandas Series to numpy array and reshape to 2D
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)

        y_train_one_hot = encoder.fit_transform(y_train)
        y_test_one_hot = encoder.transform(y_test)
        return y_train_one_hot, y_test_one_hot

    def evaluate_predictions(self, y_test, y_predictions):
        #y_test or y_predictions are 1 hot encoded, use argmax to convert to class indices
        if y_test.ndim > 1: #n dimensions. if the dimension is more than 1 use argmax
            y_test = np.argmax(y_test, axis=1)
        if y_predictions.ndim > 1:
            y_predictions = np.argmax(y_predictions, axis=1)

        # init confusion matrix for the 3 classes
        confusion_matrix = np.zeros((3, 3), dtype=int)



        # update confusion matrix with np.add.at
        np.add.at(confusion_matrix, (y_test, y_predictions), 1)

        print("Confusion Matrix:")
        print(confusion_matrix)

        # calculate accuracy
        correct_predictions = np.trace(confusion_matrix)  # TP for all classes
        total_predictions = np.sum(confusion_matrix)
        accuracy = correct_predictions / total_predictions

        self.gui.accuracy_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")
        self.plot_confusion_matrix(confusion_matrix)


    def plot_confusion_matrix(self, confusion_matrix):
        # clear the plot
        self.gui.ax1.clear()

        # plot the confusion matrix using a heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="magma",
                    xticklabels=self.classes,
                    yticklabels=self.classes, ax=self.gui.ax1, cbar=False)

        # Set titles and labels
        self.gui.ax1.set_title("Confusion Matrix")
        self.gui.ax1.set_xlabel("Predicted Class")
        self.gui.ax1.set_ylabel("Actual Class")

        # Draw the canvas
        self.gui.canvas.draw()