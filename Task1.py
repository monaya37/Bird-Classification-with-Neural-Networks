import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder    
import matplotlib.pyplot as plt

class Task1:
    def __init__(self, parent):
        self.parent = parent

        # Create main frame
        self.frame = tk.Frame(self.parent)
        self.frame.grid(padx=10, pady=10)

        # Frame for Features
        self.features_frame = tk.Frame(self.frame)
        self.features_frame.grid(row=0, column=0, padx=20)

        self.label_features = tk.Label(self.features_frame, text="Select 2 Features:")
        self.label_features.pack(anchor='w')

        self.features = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]

        self.features_list = []
        for feature in self.features:
            var = tk.IntVar()  # Create an IntVar for each option
            self.features_list.append(var)
            checkbutton = tk.Checkbutton(self.features_frame, text=feature, variable=var)
            checkbutton.pack(anchor='w')

        # Frame for Classes
        self.classes_frame = tk.Frame(self.frame)
        self.classes_frame.grid(row=0, column=1, padx=20)

        self.label_classes = tk.Label(self.classes_frame, text="Select 2 Classes:")
        self.label_classes.pack(anchor='w')

        self.classes = ["A", "B", "C"]

        self.classes_list = []
        for c in self.classes:
            var = tk.IntVar()  # Create an IntVar for each option
            self.classes_list.append(var)
            checkbutton = tk.Checkbutton(self.classes_frame, text=c, variable=var)
            checkbutton.pack(anchor='w')

        # Frame for textboxes
        self.textboxes_frame = tk.Frame(self.frame)
        self.textboxes_frame.grid(row=0, column=2, padx=20)

        # Epochs
        self.label_epochs = tk.Label(self.textboxes_frame, text="Epochs:")
        self.label_epochs.pack(anchor='w')
        self.epochs_entry = tk.Entry(self.textboxes_frame)
        self.epochs_entry.pack(anchor='w')

        # Learning Rate
        self.label_learning_rate = tk.Label(self.textboxes_frame, text="Learning Rate:")
        self.label_learning_rate.pack(anchor='w')
        self.learning_rate_entry = tk.Entry(self.textboxes_frame)
        self.learning_rate_entry.pack(anchor='w')

        # Threshold
        self.label_threshold = tk.Label(self.textboxes_frame, text="Threshold:")
        self.label_threshold.pack(anchor='w')
        self.threshold_entry = tk.Entry(self.textboxes_frame)
        self.threshold_entry.pack(anchor='w')

        # Bias Checkbox
        self.bias_var = tk.IntVar()
        self.bias_checkbox = tk.Checkbutton(self.textboxes_frame, text="Include Bias", variable=self.bias_var)
        self.bias_checkbox.pack(anchor='w', pady=(10, 10))

        # Frame for algorthims
        self.algorithms_frame = tk.Frame(self.frame)
        self.algorithms_frame.grid(row=0, column=3, padx=20)

        # Algorithm Type Radio Buttons
        self.algorithm_var = tk.StringVar(value="Algorithm1")
        self.label_algorithm = tk.Label(self.algorithms_frame, text="Algorithm Type:")
        self.label_algorithm.pack(anchor='w')

        self.radio_algorithm1 = tk.Radiobutton(self.algorithms_frame, text="Perceptron ", variable=self.algorithm_var, value="Algorithm1")
        self.radio_algorithm2 = tk.Radiobutton(self.algorithms_frame, text="Adaline", variable=self.algorithm_var, value="Algorithm2")
        self.radio_algorithm1.pack(anchor='w')
        self.radio_algorithm2.pack(anchor='w')

        # Button to show selected options
        self.submit_button = tk.Button(self.frame, text="Submit", command=self.run_algorithm)
        self.submit_button.grid(row=1, column=0, columnspan=4, pady=20)

        self.dataset = None


    #Getters
    def get_epochs(self):
        return self.epochs_entry.get()

    def get_learning_rate(self):
        return self.learning_rate_entry.get()

    def get_threshold(self):
        return self.threshold_entry.get()

    def get_chosen_features(self):
        return [feature for feature, var in zip(self.features, self.features_list) if var.get()]

    def get_chosen_classes(self):
        return [c for c, var in zip(self.classes, self.classes_list) if var.get()]

    def get_bias_state(self):
        return self.bias_var.get()
    
    def get_algorithm_type(self):
        return self.algorithm_var.get()
    
    
    #Functions
    def run_algorithm (self):
        dataset = self.read_file('birds.csv')
        X_train, X_test, y_train, y_test = self.split_to_train_test(dataset)
        X_train, X_test ,y_train, y_test = self.preprocess_data(X_train, X_test, y_train, y_test)

        algorithm_type = self.get_algorithm_type()

        if(algorithm_type == "Algorithm1"):
            self.perceptron()
        else:
            self.adaline()


    def read_file(self, path):
        dataset = pd.read_csv(path)
        print(dataset.head())
        return dataset


    def get_classes(dataset, selected_classes):
        class_data = []
        for cls in selected_classes:
            class_data.append(dataset[dataset['category'] == cls])
        return pd.concat(class_data)


    def perceptron(self):
        #implement perceptron
        #get values and implment
        return
    

    #pass preprocessed X
    def adaline(self, X, y_actual):

        epochs = self.get_epochs()
        learning_rate = self.get_learning_rate()
        threshold = self.get_threshold()
        num_of_features = self.get_chosen_features().count()
        include_bias = self.get_bias_state()

        weights = np.ones(num_of_features)
        bias = 0
        errors = []
        y_predict = []



        if(include_bias):

            for e in range(epochs):
                for i in range(X.shape[0]):

                    y_predict = sum(weights * X[i]) + bias
                    error = y_actual[i] -  y_predict
                
                    # update weights and bias
                    weights = weights + learning_rate * error * X[i]
                    bias = bias + learning_rate * error

                    errors.append((error ** 2))
                
                mse = np.mean(errors)
                if mse < threshold: #from HUI
                    break

        else:

            for e in range(epochs):

                for i in range(X.shape[0]):

                    y_predict = sum(weights * X[i]) 
                    error = y_actual[i] -  y_predict
                
                    # update weights and bias
                    weights = weights + learning_rate * error * X[i]
                    errors.append((error ** 2))
                
                mse = np.mean(errors)
                if mse < threshold: #from HUI
                    break


    def signum_funcation(self, x):
        if(x > 0):
            return 1
        if(x < 0):
            return -1
        else:
            return 0

    def split_to_train_test(self, dataset):
        
        selected_features = self.get_chosen_features()
        selected_classes = self.get_chosen_classes()

        filtered_data = self.get_classes(dataset, selected_classes)

        # Select only the chosen features along with the class
        X = filtered_data[selected_features]
        y = filtered_data['category']

        # Train-test split (80-20 ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        return X_train, X_test, y_train, y_test


    def plot_function(self, X, bias, weights):

        include_bias = self.get_bias_state()
        features = self.get_chosen_features()

        x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)

        if(include_bias):
            # Generate points for the decision boundary line
            x2 = -(weights[0] * x1 + bias) / weights[1]
        else:
            x2 = -(weights[0] * x1)


        # Plot the decision boundary line
        plt.plot(x1, x2, color='green', label='Decision Boundary')

        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend()
        plt.show() 


    def show_selected(self):
        print("Selected Options", 
                        f"Features: {self.get_chosen_features()}\n"
                        f"Classes: {self.get_chosen_classes()}\n"
                        f"Learning Rate: {self.get_learning_rate()}\n"
                        f"Threshold: {self.get_threshold()}\n"
                        f"Include Bias: {self.bias_var.get()}\n"
                        f"Algorithm Type: {self.get_algorithm_type()}")
    


    #GUI Functions
    def display(self):
        self.frame.grid(row=1, column=0, columnspan=2, sticky='nsew')

    def hide(self):
        self.frame.grid_forget()