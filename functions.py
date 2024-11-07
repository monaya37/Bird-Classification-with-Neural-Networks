    
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder    
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Getters

class functions:

    def __init__(self, gui):
        self.gui = gui

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
        dataset = self.read_file('birds.csv')

        #this function retruns x and y based on user selection
        X_train, X_test, y_train, y_test = self.split_to_train_test(dataset)

        X_train, X_test = self.preprocess(X_train, X_test, self.get_chosen_features())
        y_train, y_test= self.preprocess_y(y_train, y_test)


        algorithm_type = self.get_algorithm_type()

        if(algorithm_type == "Algorithm1"):
            #TO-DO:
            weights, bias = self.perceptron(X_train, y_train)
            y_pred = self.predict(X_test, weights, bias)
            self.plot_function(X_train, y_train, weights, bias)


        else:
            weights, bias = self.adaline(X_train, y_train)
            #y_pred = self.predict(X_test, weights, bias)
            self.plot_function(X_train, y_train, weights, bias)



    def read_file(self, path):
        dataset = pd.read_csv(path)
        print(dataset.head())
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


        numeric_cols = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']
        numeric_cols = list(set(numeric_cols) & set(selected_features))

        lower_bounds = {}
        upper_bounds = {}

        for col in numeric_cols:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bounds[col] = Q1 - 1.5 * IQR
            upper_bounds[col] = Q3 + 1.5 * IQR

            median = X_train[col].median()
            X_train[col] = X_train[col].apply(lambda x: x if lower_bounds[col] <= x <= upper_bounds[col] else median)
        
        for col in numeric_cols:
            X_test[col] = X_test[col].apply(
                lambda x: x if lower_bounds[col] <= x <= upper_bounds[col] else X_train[col].median())
            
        #Label Enoding
        le = LabelEncoder()
        for i in X_train.columns: 
            if (X_train[i].dtype == "O"):  # if the column is categorical
                le.fit(X_train[i])  
                X_train[i] = le.transform(X_train[i]) 
                X_test[i] = le.transform(X_test[i]) 


        # Normalizing
        scaler = StandardScaler()
        col_names= X_train.columns
        scaler.fit(X_train[col_names])
        X_train[col_names] = scaler.transform(X_train[col_names])
        X_test[col_names] = scaler.transform(X_test[col_names])
            
        return X_train, X_test
    

    def preprocess_y(self, y_train, y_test):

        le = LabelEncoder()

        if (y_train.dtype == "O"):  
            le.fit(y_train)  
            y_train = le.transform(y_train)  
            y_test = le.transform(y_test)  

        return y_train, y_test


    def perceptron(self, X, y_actual):
        
        X = X.values
        y_actual = np.array(y_actual, dtype=float)

        epochs = int(self.get_epochs())
        learning_rate = float(self.get_learning_rate())
        include_bias = self.get_bias_state()

        weights = np.random.randn(X.shape[1])
        bias = np.random.randn(1) / 2

        y_predict = []
        counter = 0

        if(include_bias):

            for e in range(epochs):
                counter = 0
                for i in range(X.shape[0]):
                    
                    y_predict = sum(weights * X[i]) + bias
                    prediction = self.signum(y_predict)
                    if(y_actual[i] != prediction):
                        # update weights and bias
                        counter = 0
                        weights = weights + learning_rate * X[i]
                        bias = bias + learning_rate 
                    else:
                        counter += 1

                if(counter == X.shape[0]):
                    break              

        else:
            for e in range(epochs):
                counter = 0
                for i in range(X.shape[0]):

                    y_predict = sum(weights * X[i]) 
                    prediction = self.signum(y_predict)
                
                    if(y_actual[i] != prediction):
                        counter = 0
                        weights = weights + learning_rate * X[i]  
                    else:
                        counter += 1

                if(counter == X.shape[0]):
                    break              

        return weights, bias
        


    def adaline(self, X, y_actual):

        X = X.values
        y_actual = np.array(y_actual, dtype=float)

        epochs = int(self.get_epochs())
        learning_rate = float(self.get_learning_rate())
        threshold = float(self.get_threshold())
        include_bias = self.get_bias_state()

        weights = np.random.randn(X.shape[1])
        bias = np.random.randn(1) / 2

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

        return weights, bias


    def signum(self, x):
        if(x > 0):
            return 1
        if(x < 0):
            return -1
        else:
            return 0
        
    def predict(self, X_test, weights, bias):

        X_test = X_test.values

        include_bias = self.get_bias_state()
        predictions = []

        if(include_bias):
            for i in range(X_test.shape[0]):
                y_predict = np.dot(weights, X_test[i]) + bias
                predictions.append(y_predict)
                
        else:
            for i in range(X_test.shape[0]):
                y_predict = np.dot(weights, X_test[i])
                predictions.append(y_predict)

        return np.array(predictions)

    def split_to_train_test(self, dataset):
        
        selected_features = self.get_chosen_features()
        selected_classes = self.get_chosen_classes()

        filtered_data = self.get_classes(dataset, selected_classes)

        # Select only the chosen features along with the class
        X = filtered_data[selected_features]
        y = filtered_data['bird category']


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)


        return X_train, X_test, y_train, y_test


    def plot_function(self, X, Y, weights, bias):

        include_bias = self.get_bias_state()
        features = self.get_chosen_features()

        X = X.values

        # Plot the data points    
        plt.scatter(X[Y == 0, 0], X[Y == 0, 1], color='red', label='Class 0')
        plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='blue', label='Class 1')

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