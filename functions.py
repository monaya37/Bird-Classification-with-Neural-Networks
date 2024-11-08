import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder    
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from random import shuffle
from sklearn.utils import shuffle
import seaborn as sns

#Getters
class functions:

    def __init__(self, gui):
        self.gui = gui

        self.epochs = None
        self.learning_rate = None
        self.threshold = None
        self.selected_features = None
        self.selected_classes = None
        self.include_bias = None
        self.algorithm_type = None
        self.dataset = None

    #Getters
    def get_epochs(self):
        return self.gui.epochs_entry.get()

    def get_learning_rate(self):
        return self.gui.learning_rate_entry.get()

    def get_threshold(self):
        return self.gui.threshold_entry.get()

    def get_chosen_features(self):
        return [feature for feature, var in zip(self.gui.features, self.gui.features_list) if var.get()]

    def get_chosen_classes(self):
        return [c for c, var in zip(self.gui.classes, self.gui.classes_list) if var.get()]

    def get_bias_state(self):
        return self.gui.bias_var.get()

    def get_algorithm_type(self):
        return self.gui.algorithm_var.get()

    
    #Functions
    def run_algorithm(self):

        self.epochs = int(self.get_epochs())
        self.learning_rate = float(self.get_learning_rate())
        self.threshold = float(self.get_threshold())
        self.selected_features = self.get_chosen_features()
        self.selected_classes = self.get_chosen_classes()
        self.include_bias = self.get_bias_state()
        self.algorithm_type = self.get_algorithm_type()

        # Read dataset and split it based on chosen features and classes
        X_train, X_test, y_train, y_test = self.split_to_train_test(self.read_file('birds.csv'))

        # Preprocess training and test sets
        X_train, X_test = self.preprocess(X_train, X_test, self.get_chosen_features())
        y_train, y_test = self.preprocess_y(y_train, y_test)

        # Determine which algorithm to run
        if self.algorithm_type == "Algorithm1":  # Perceptron
            weights, bias = self.perceptron(X_train, y_train)
            y_pred = self.predict(X_test, weights, bias)
            prediction = np.array(self.signum(y_pred))  # Apply signum to make predictions discrete
            print(prediction)
            # Evaluate the performance of the model using confusion matrix and accuracy
            self.evaluate_predictions(y_test, prediction)  # Evaluate on y_test
            self.plot_function(X_train, y_train, weights, bias)

        else:  # Adaline
            weights, bias = self.adaline(X_train, y_train)
            y_pred = self.predict(X_test, weights, bias)
            prediction = np.array(y_pred)
            # Apply threshold element-wise to each prediction
            y_predict_class = (prediction >= 0).astype(float)
            print(y_predict_class)
            # Evaluate the performance of the model using confusion matrix and accuracy
            self.evaluate_predictions_adaline(y_test, y_predict_class)  #Evaluate on y_test
            self.plot_function(X_train, y_train, weights, bias)


    def read_file(self, path):
        dataset = pd.read_csv(path)
        return dataset


    def get_classes(self, dataset, selected_classes):
        class_data = []
        for cls in selected_classes:
            class_data.append(dataset[dataset['bird category'] == cls])
        return pd.concat(class_data)

    def preprocess(self, X_train, X_test, selected_features):

        #filling na
        if 'gender' in X_train.columns:
            gender_mode = X_train['gender'].mode()[0]
            X_train['gender'].fillna(gender_mode, inplace=True)
            X_test['gender'].fillna(gender_mode, inplace=True)

        # Handle outliers in numerical columns using IQR for train and test set
        # Select numeric columns that are both in the dataset and in selected_features
        numeric_cols = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']
        numeric_cols = list(set(numeric_cols) & set(selected_features))

        # Calculate the IQR and bounds using X_train
        lower_bounds = {}
        upper_bounds = {}

        for col in numeric_cols:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bounds[col] = Q1 - 1.5 * IQR
            upper_bounds[col] = Q3 + 1.5 * IQR

            # Apply outlier handling on X_train by replacing outliers with median
            median = X_train[col].median()
            X_train[col] = X_train[col].apply(lambda x: x if lower_bounds[col] <= x <= upper_bounds[col] else median)
        
        # Apply the same outlier handling to X_test
        for col in numeric_cols:
            # Apply the same bounds calculated from the X_train set to X_test
            X_test[col] = X_test[col].apply(
                lambda x: x if lower_bounds[col] <= x <= upper_bounds[col] else X_train[col].median())
            
        # Create a LabelEncoder object
        le = LabelEncoder()
        # Loop through each column and label encode categorical columns
        for i in X_train.columns: #iterate over all columns
            if (X_train[i].dtype == "O"):  # if the column is categorical
                le.fit(X_train[i])  # fit on the training data
                X_train[i] = le.transform(X_train[i])  # transform the training data
                X_test[i] = le.transform(X_test[i])  # transform the test data

        # A--> 0, B--> 1, C-->2       (el encoding)
        scaler = StandardScaler()
        col_names= X_train.columns
        scaler.fit(X_train[col_names])
        X_train[col_names] = scaler.transform(X_train[col_names])
        X_test[col_names] = scaler.transform(X_test[col_names])
            
        return X_train, X_test
    
    def preprocess_y(self, y_train, y_test):

        le = LabelEncoder()
        if (y_train.dtype == "O"):  # if the column is categorical
            le.fit(y_train)  
            y_train = le.transform(y_train)  
            y_test = le.transform(y_test) 
    
        return y_train, y_test

    def perceptron(self, X, y_actual):

        X = X.values
        y_actual = np.array(y_actual, dtype=float)


        weights = np.random.randn(X.shape[1])
        bias = np.random.randn(1) / 2
        y_predict = []
        counter = 0

        if (self.include_bias):

            for e in range(self.epochs):
                counter = 0
                for i in range(X.shape[0]):
                    y_predict = sum(weights * X[i]) + bias
                    prediction = self.signum(y_predict)
                    if (y_actual[i] != prediction):
                        # update weights and bias
                        counter = 0
                        weights = weights + self.learning_rate * X[i]
                        bias = bias + self.learning_rate
                    else:
                        counter += 1

                if (counter == X.shape[0]):
                    break

        else:

            for e in range(self.epochs):
                counter = 0
                for i in range(X.shape[0]):
                    y_predict = sum(weights * X[i])
                    prediction = self.signum(y_predict)
                    error = y_actual[i] - prediction

                    if (y_actual[i] != prediction):
                        counter = 0
                        weights = weights + self.learning_rate * X[i]
                    else:
                        counter += 1

                if (counter == (X.shape[0])):
                    break

        return weights, bias

    def adaline(self, X, y_actual):
        y_actual = np.array(y_actual, dtype=float)
        X = X.values


        num_of_features = len(self.get_chosen_features())

        weights = np.random.randn(num_of_features)
        bias = np.random.randn(1) / 2
        errors = []
        y_predict = []

        print("x type is: ", type(X))

        if(self.include_bias):
            for e in range(self.epochs):
                for i in range(X.shape[0]):
                    y_predict = sum(weights * X[i]) + bias
                    error = y_actual[i] -  y_predict

                    # update weights and bias
                    weights = weights + self.learning_rate * error * X[i]
                    bias = bias + self.learning_rate * error

                    errors.append((error ** 2))

                mse = np.mean(errors)
                if mse < self.threshold: #from GUI
                    break

        else:
            for e in range(self.epochs):
                for i in range(X.shape[0]):
                    y_predict = sum(weights * X[i])
                    error = y_actual[i] -  y_predict

                    # update weights and bias
                    weights = weights + self.learning_rate * error * X[i]
                    errors.append((error ** 2))

                mse = np.mean(errors)
                if mse < self.threshold: #from GUI
                    break

        return weights, bias

    def signum(self, x):
        return np.where(x >= 0, 1, -1)  # Converts all values to 1 or -1

    def predict(self, X_test, weights, bias):

        predictions = []
        X_test = X_test.values

        if(self.include_bias):
            for i in range(X_test.shape[0]):
                y_predict = np.dot(weights, X_test[i]) + bias
                predictions.append(y_predict)
        else:
            for i in range(X_test.shape[0]):
                y_predict = np.dot(weights, X_test[i])
                predictions.append(y_predict)

        return np.array(predictions)

    def evaluate_predictions(self, y_true, y_pred):
        TP = sum((y_true == 1) & (y_pred == 1))
        TN = sum((y_true == -1) & (y_pred == -1))
        FP = sum((y_true == -1) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == -1))

        # Avoid division by zero
        if np.all(TP + TN + FP + FN == 0):
            print("Warning: No predictions made.")
            accuracy = 0
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)

        print(f"Confusion Matrix:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Accuracy: {accuracy}")



    def evaluate_predictions_adaline(self, y_true, y_pred):
        TP = sum((y_true == 1) & (y_pred == 1))
        TN = sum((y_true == 0) & (y_pred == 0))
        FP = sum((y_true == 0) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))

        # Avoid division by zero
        if np.all(TP + TN + FP + FN == 0):
            print("Warning: No predictions made.")
            accuracy = 0
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)

        print(f"Confusion Matrix:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Accuracy: {accuracy}")




    def split_to_train_test(self, dataset):


        # Split each class data into 30 samples for training and 20 for testing
        train_data = []
        test_data = []
        for cls in self.selected_classes:
            class_data = dataset[dataset['bird category'] == cls]
            class_data = shuffle(class_data, random_state =0)  # Shuffle data within each class
            print(cls)

            # Select 30 samples for training and 20 for testing
            train_data.append(class_data.iloc[:30])
            test_data.append(class_data.iloc[30:])

        # Concatenate data for training and testing
        train_data = pd.concat(train_data)
        test_data = pd.concat(test_data)

        # Separate features and labels
        X_train = train_data[self.selected_features]
        y_train = train_data['bird category']
        X_test = test_data[self.selected_features]
        y_test = test_data['bird category']

        return X_train, X_test, y_train, y_test

    def plot_function(self, X, Y, weights, bias):

        X = X.values
       
        self.gui.ax.clear()
        # Plot the data points    
        self.gui.ax.scatter(X[Y == 0, 0], X[Y == 0, 1], color='red', label='Class 0')
        self.gui.ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='blue', label='Class 1')

        x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)

        if(self.include_bias):
            # Generate points for the decision boundary line
            x2 = -(weights[0] * x1 + bias) / weights[1]
        else:
            x2 = -(weights[0] * x1)


        # Plot the decision boundary line
        self.gui.ax.plot(x1, x2, color='green', label='Decision Boundary')

        self.gui.ax.set_xlabel(self.selected_features[0])
        self.gui.ax.set_ylabel(self.selected_features[1])
        self.gui.ax.legend()
        self.gui.canvas.draw()

