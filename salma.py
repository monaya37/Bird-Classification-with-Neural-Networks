import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from random import shuffle
from sklearn.utils import shuffle
import seaborn as sns
#from Functions import *
import math


# task2 functions
class Task2Functions:

    def _init_(self, gui):
        self.gui = gui

        self.epochs = None
        self.learning_rate = None
        self.include_bias = None
        self.activation_function = None  # Store activation function
        self.dataset = None
        self.classes = None
        self.features = None
        self.min = None
        self.max = None
        self.weights = None
        self.bias = None





   
    def preprocess_target(self, y_train, y_test):
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)  # sparse=False ensures that the output is a dense array

        # Convert pandas Series to numpy array and reshape to 2D
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)

        # Fit the encoder and transform the labels
        y_train_one_hot = encoder.fit_transform(y_train)
        y_test_one_hot = encoder.transform(y_test)
        print(y_test_one_hot)
        return y_train_one_hot, y_test_one_hot

    def run_algorithm(self):
        """
        Main entry point for running the algorithm. Dynamically fetches required parameters and processes the dataset.
        """
        #initialize global variables
        self.dataset = self.read_file('birds.csv')
        self.epochs = int(self.gui.get_epochs())
        self.learning_rate = float(self.gui.get_learning_rate())
        self.include_bias = self.gui.get_bias_state()
        self.activation_function = self.gui.get_algorithm_type()
        self.num_hidden_layers = int(self.gui.get_num_of_hidden_layers())
        self.hidden_layers = self.gui.get_hidden_layers()
        self.num_of_neurons = int(self.gui.get_num_of_neurons())
        self.neurons = self.gui.get_neurons()
        self.gui.show_selected()
        self.selected_classes = ['A', 'B', 'C']
        self.selected_features = ['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']

        # Split based on selected
        X_train, X_test, y_train, y_test = split_to_train_test(self.dataset, self.selected_classes, self.selected_features)

        print("el classes", self.selected_classes)
        # preprocess
        X_train, X_test = preprocess_features(X_train, X_test, self.selected_features)
        y_train, y_test = self.preprocess_target(y_train, y_test)
        print("el yyyyyclasses", len(y_train))

        # define input and output size
        input_size = X_train.shape[1]
        # find the class labels (the index of the '1' in each one-hot encoded vector)
        class_labels = np.argmax(y_train, axis=1)
        unique_classes = np.unique(class_labels)
        output_size = len(unique_classes)

        print("output size",output_size)

        #output_size = len(np.unique(y_train))  # number of classes
        #print("output size", output_size)

        #train
        weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = self.train_model(
            X_train, y_train, input_size, self.num_hidden_layers, output_size,
            self.learning_rate, self.epochs, self.activation_function)

        # test
        y_test_predictions = self.predict_neural_network(X_test, weights_input_hidden, weights_hidden_output,
                                                         bias_hidden, bias_output, self.activation_function)
        # confusion metrics
        self.evaluate_predictions(y_test, y_test_predictions)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def initialize_weights(self, input_size, hidden_size, output_size):


        #.uniform function generates random numbers from a uniform distribution.
        weights_input_hidden = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        weights_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        bias_hidden = np.zeros((1, hidden_size)) # 1 bias term per neuron
        bias_output = np.zeros((1, output_size))
        return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # to avoid overflow
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward_pass(self, X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output,
                     activation_function):
        # hidden layer
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden

        # apply activ function for  hidden layer
        if activation_function == "Sigmoid":
            hidden_output = self.sigmoid(hidden_input)
        elif activation_function == "Tanh":
            hidden_output = self.tanh(hidden_input)

        # output layer
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output

        # apply softmax 4 probabilities(since it is not binary classif)
        final_output = self.softmax(final_input)

        return hidden_input, hidden_output, final_input, final_output

    def backward_pass(self, X, y, hidden_input, hidden_output, final_output, weights_hidden_output, learning_rate,
                      activation_function):

        # output layer error and gradient
        print("shape of y", y.shape)
        print("shape of final out", final_output.shape)

        output_error = y - final_output
        output_gradient = output_error * self.sigmoid_derivative(final_output)  # Output uses sigmoid

        # hidden layer error and GD
        hidden_error = np.dot(output_gradient, weights_hidden_output.T)
        if activation_function == "Sigmoid":
            hidden_gradient = hidden_error * self.sigmoid_derivative(hidden_output)
        elif activation_function == "Tanh":
            hidden_gradient = hidden_error * self.tanh_derivative(hidden_output)

        # updates for w and b
        weights_hidden_output_update = np.dot(hidden_output.T, output_gradient)
        weights_input_hidden_update = np.dot(X.T, hidden_gradient)
        bias_output_update = np.sum(output_gradient, axis=0, keepdims=True)
        bias_hidden_update = np.sum(hidden_gradient, axis=0, keepdims=True)

        return weights_input_hidden_update, weights_hidden_output_update, bias_hidden_update, bias_output_update
    def train_model(self, X_train, y_train, input_size, hidden_size, output_size, learning_rate,
                    epochs, activation_function):
        # init w and b
        weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = self.initialize_weights(
            input_size, hidden_size, output_size)

        for epoch in range(epochs):
            # forward prop
            hidden_input, hidden_output, final_input, final_output = self.forward_pass(
                X_train, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, activation_function)

            # back prop
            weights_input_hidden_update, weights_hidden_output_update, bias_hidden_update, bias_output_update = self.backward_pass(
                X_train, y_train, hidden_input, hidden_output, final_output, weights_hidden_output, learning_rate, activation_function)

            # update w and b after every epoch
            weights_input_hidden += learning_rate * weights_input_hidden_update
            weights_hidden_output += learning_rate * weights_hidden_output_update
            bias_hidden += learning_rate * bias_hidden_update
            bias_output += learning_rate * bias_output_update

        return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    def predict_neural_network(self, X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output,
                               activation_function):
        # forward prop 4 predictions
        _, _, _, final_output = self.forward_pass(X, weights_input_hidden, weights_hidden_output, bias_hidden,
                                                  bias_output, activation_function)

        # index of the class with the highest prob
        return np.argmax(final_output, axis=1)

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
        print("accuracy:", accuracy)

        self.gui.accuracy_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")
        self.plot_confusion_matrix(confusion_matrix)


    def plot_confusion_matrix(self, confusion_matrix):
        # clear the plot
        self.gui.ax2.clear()

        # plot the confusion matrix using a heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="magma",
                    xticklabels=self.selected_classes,
                    yticklabels=self.selected_classes, ax=self.gui.ax2, cbar=False)

        # Set titles and labels
        self.gui.ax2.set_title("Confusion Matrix")
        self.gui.ax2.set_xlabel("Predicted Class")
        self.gui.ax2.set_ylabel("Actual Class")

        # Draw the canvas
        self.gui.canvas.draw()