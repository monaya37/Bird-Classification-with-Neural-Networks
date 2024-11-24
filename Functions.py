
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder    
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from random import shuffle
from sklearn.utils import shuffle
import seaborn as sns

def read_file(path):
    return pd.read_csv(path)

def preprocess_features(X_train, X_test, selected_features):

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

    X_train = X_train.values
    return X_train, X_test

def preprocess_target(y_train, y_test):

    le = LabelEncoder()
    if (y_train.dtype == "O"):  # if the column is categorical
        le.fit(y_train)  
        y_train = le.transform(y_train)  
        y_test = le.transform(y_test) 

    return y_train, y_test
