import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder    
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from random import shuffle
from sklearn.utils import shuffle
import seaborn as sns

# task1 functions
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
        self.activation_function = None
        self.dataset = None

    # Functions
    def run_algorithm(self):

        #initilize global variables
        self.dataset = self.read_file('birds.csv')
        self.epochs = int(self.gui.get_epochs())
        self.learning_rate = float(self.gui.get_learning_rate())
        self.threshold = float(self.gui.get_threshold())
        self.selected_features = self.gui.get_selected_features()
        self.selected_classes = self.gui.get_selected_classes()
        self.include_bias = self.gui.get_bias_state()
        self.algorithm_type = self.gui.get_algorithm_type()

        if(self.algorithm_type == 'Perceptron'):
            self.activation_function = self.signum
            #min = -1
            min = 0
            max = 1
        else:
            self.activation_function = self.linear
            min = 0
            max = 1

        # Split based on selected
        X_train, X_test, y_train, y_test = self.split_to_train_test(self.dataset)

        # Preprocess
        X_train, X_test = self.preprocess_features(X_train, X_test, self.selected_features)
        y_train, y_test = self.preprocess_target(y_train, y_test)

        # Train model
        weights, bias = self.train_model(X_train, y_train)
        y_pred = self.predict(X_test, weights, bias)
        print("y predcted: ", y_pred)
        print("y actual: ", y_test)

        # Evaluate the performance
        TP , TN, FP, FN = self.evaluate_predictions(y_test, y_pred, min, max) 
        self.plot_confusion_matrix(TP, TN, FP, FN)

        self.plot_function(X_train, y_train, weights, bias)


    def read_file(self, path):
        dataset = pd.read_csv(path)
        return dataset


    def preprocess_features(self, X_train, X_test, selected_features):

        # Fill na
        if "gender" in X_train.columns:
            gender_mode = X_train["gender"].mode()[0]
            X_train["gender"].fillna(gender_mode, inplace=True)
            X_test["gender"].fillna(gender_mode, inplace=True)

        # Handle outliers
        numeric_cols = ["body_mass", "beak_length", "beak_depth", "fin_length"]
        numeric_cols = list(set(numeric_cols) & set(selected_features))

        lower_bounds = {}
        upper_bounds = {}

        for col in numeric_cols:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bounds[col] = Q1 - 1.5 * IQR
            upper_bounds[col] = Q3 + 1.5 * IQR

            # Replace outliers with median
            median = X_train[col].median()
            X_train[col] = X_train[col].apply(
                lambda x: x if lower_bounds[col] <= x <= upper_bounds[col] else median
            )

        # handle outliears (X_test)
        for col in numeric_cols:
            X_test[col] = X_test[col].apply(
                lambda x: (
                    x
                    if lower_bounds[col] <= x <= upper_bounds[col]
                    else X_train[col].median()
                )
            )

        # Encode categoricals
        le = LabelEncoder()
        for i in X_train.columns:
            if X_train[i].dtype == "O":  # if the column is categorical
                le.fit(X_train[i])
                X_train[i] = le.transform(X_train[i])
                X_test[i] = le.transform(X_test[i])


        # Scale data
        scaler = StandardScaler()
        col_names = X_train.columns
        scaler.fit(X_train[col_names])
        X_train[col_names] = scaler.transform(X_train[col_names])
        X_test[col_names] = scaler.transform(X_test[col_names])

        return X_train, X_test


    def preprocess_target(self, y_train, y_test):

        le = LabelEncoder()
        if (y_train.dtype == "O"):  # if the column is categorical
            le.fit(y_train)  
            y_train = le.transform(y_train)  
            y_test = le.transform(y_test) 

        return y_train, y_test


    def train_model(self, X_train, y_train):

        n = X_train.shape[0]
        X_train = X_train.values

        y_train = np.array(y_train, dtype=float)
        weights = np.random.randn(X_train.shape[1])

        bias = 0

        
        if self.algorithm_type == "Perceptron":            
            for _ in range(self.epochs):
                counter = 0

                for i in range(n):
                    y_predict = sum(weights * X_train[i]) + bias
                    y_predict = self.activation_function(y_predict)
                    error = y_train[i] -  y_predict
                    if error != 0:
                        counter = 0
                        weights, bias = self.update_weights_and_bias(
                            weights, bias, X_train[i], error
                        )
                    else:
                        counter += 1

                if counter == n:
                    break

        elif(self.algorithm_type == 'Adaline'):

            for _ in range(self.epochs):
                
                errors = []

                for i in range(n):

                    y_predict = sum(weights * X_train[i]) + bias
                    y_predict = self.activation_function(y_predict)
                    error = y_train[i] -  y_predict

                    weights, bias = self.update_weights_and_bias(
                        weights, bias, X_train[i], error
                    )

                    errors.append((error**2))

                mse = np.mean(errors)
                if mse < self.threshold:
                    break

        return weights, bias


    def update_weights_and_bias(self, weights, bias, X, error):

        if self.include_bias:
            weights = weights + self.learning_rate * error * X
            bias = bias + self.learning_rate * error
        else:
            weights = weights + self.learning_rate * error * X

        return weights, bias


    def signum(self, x):
           return np.where(x >= 0, 1, -1) 


    def linear(self, x):
        return float(x) 


    def predict(self, X_test, weights, bias):

        predictions = []
        X_test = X_test.values

        if(self.include_bias):
            for i in range(X_test.shape[0]):
                y_predict = np.dot(weights, X_test[i]) + bias
                y_predict = self.activation_function(y_predict)
                predictions.append(y_predict)
        else:
            for i in range(X_test.shape[0]):
                y_predict = np.dot(weights, X_test[i])
                y_predict = self.activation_function(y_predict)
                predictions.append(y_predict)

        predictions = np.array(predictions)

        if(self.algorithm_type == 'Perceptron'):
            predictions = np.array([1 if prediction == 1 else 0 for prediction in predictions])

        if(self.algorithm_type == 'Adaline'):
            predictions = (predictions >= 0.5).astype(int)

        return predictions


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

        print(f"Confusion Matrix:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Accuracy: {accuracy}%")

        return TP, TN, FP, FN 


    def split_to_train_test(self, dataset):

        train_data = []
        test_data = []

        for cls in self.selected_classes:

            class_data = dataset[dataset['bird category'] == cls]
            class_data = shuffle(class_data, random_state =0)  # Shuffle data within each class
            #print(cls)

            # Select 30 samples for training and 20 for testing
            train_data.append(class_data.iloc[:30])
            test_data.append(class_data.iloc[30:])

        train_data = pd.concat(train_data)
        test_data = pd.concat(test_data)

        X_train = train_data[self.selected_features]
        y_train = train_data['bird category']
        X_test = test_data[self.selected_features]
        y_test = test_data['bird category']

        return X_train, X_test, y_train, y_test


    def plot_function(self, X, Y, weights, bias):

        X = X.values

        self.gui.ax1.clear()
        self.gui.ax1.scatter(X[Y == 0, 0], X[Y == 0, 1], color='red', label=self.selected_classes[0])
        self.gui.ax1.scatter(X[Y == 1, 0], X[Y == 1, 1], color='blue', label=self.selected_classes[1])

        x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)

        # bias = 0, if the box is unchecked
        x2 = -(weights[0] * x1 + bias) / weights[1]


        # Plot the decision boundary line
        self.gui.ax1.plot(x1, x2, color='green', label='Decision Boundary')

        self.gui.ax1.set_xlabel(self.selected_features[0])
        self.gui.ax1.set_ylabel(self.selected_features[1])
        self.gui.ax1.legend()
        self.gui.canvas.draw()


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


    # check w salma 

    # def evaluate_predictions_perceptron(self, y_true, y_pred):
    #     TP = sum((y_true == 1) & (y_pred == 1))
    #     TN = sum((y_true == -1) & (y_pred == -1))
    #     FP = sum((y_true == -1) & (y_pred == 1))
    #     FN = sum((y_true == 1) & (y_pred == -1))

    #     # Avoid division by zero
    #     if np.all(TP + TN + FP + FN == 0):
    #         print("Warning: No predictions made.")
    #         accuracy = 0
    #     else:
    #         accuracy = (TP + TN) / (TP + TN + FP + FN)

    #     print(f"Confusion Matrix:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    #     print(f"Accuracy: {accuracy}")

    # def evaluate_predictions_adaline(self, y_true, y_pred):
    #     TP = sum((y_true == 1) & (y_pred == 1))
    #     TN = sum((y_true == 0) & (y_pred == 0))
    #     FP = sum((y_true == 0) & (y_pred == 1))
    #     FN = sum((y_true == 1) & (y_pred == 0))

    #     # Avoid division by zero
    #     if np.all(TP + TN + FP + FN == 0):
    #         print("Warning: No predictions made.")
    #         accuracy = 0
    #     else:
    #         accuracy = (TP + TN) / (TP + TN + FP + FN)

    #     print(f"Confusion Matrix:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    #     print(f"Accuracy: {accuracy}")
