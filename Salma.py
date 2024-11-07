#importing the libraries
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



#reading the csv file into a dataframe
df=pd.read_csv('birds.csv')

class_A = df[df['bird category'] == 'A']
class_B = df[df['bird category'] == 'B']
class_C = df[df['bird category'] == 'C']

# train test split before handling outliers
X_a = class_A.drop(columns=['bird category'])
y_a = class_A['bird category']
X_train_a, X_test_a, y_train_a, y_test_a = (train_test_split(X_a, y_a, test_size=0.2, shuffle=True, random_state=0))

X_b = class_B.drop(columns=['bird category'])
y_b = class_B['bird category']
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.2, shuffle=True, random_state=0)

X_c = class_C.drop(columns=['bird category'])
y_c = class_C['bird category']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, shuffle=True, random_state=0)


# Merge training sets
X_train = pd.concat([X_train_a, X_train_b, X_train_c], ignore_index=True)
y_train = pd.concat([y_train_a, y_train_b, y_train_c], ignore_index=True)

# Merge test sets
X_test = pd.concat([X_test_a, X_test_b, X_test_c], ignore_index=True)
y_test = pd.concat([y_test_a, y_test_b, y_test_c], ignore_index=True)

gender_mode = X_train['gender'].mode()[0]
X_train['gender'].fillna(gender_mode, inplace=True)
X_test['gender'].fillna(gender_mode, inplace=True)

# Handle outliers in numerical columns using IQR for train and test set
numeric_cols = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']

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


# Normalize the numerical features
scaler = StandardScaler()
# Select the features to scale
col_names= X_train.columns
# Fit the scaler on the training data (ONLY on X_train)
scaler.fit(X_train[col_names])
# Transform the training and test data using the fitted scaler
X_train[col_names] = scaler.transform(X_train[col_names])
X_test[col_names] = scaler.transform(X_test[col_names])


def submit_action():
    # Collecting values from features checkboxes
    selected_features = []
    for i, feature_var in enumerate(features_list):
        if feature_var.get() == 1:
            selected_features.append(features[i])

    # Collecting values from classes checkboxes
    selected_classes = []
    for i, class_var in enumerate(classes_list):
        if class_var.get() == 1:
            selected_classes.append(classes[i])

    # Collecting values from entry fields
    learning_rate = learning_rate_entry.get()  # This returns a string
    threshold = threshold_entry.get()  # This returns a string

    # Collecting the value of bias checkbox
    include_bias = bias_var.get()  # Returns 1 if checked, 0 if not

    # Collecting the selected algorithm type
    algorithm_type = algorithm_var.get()

    # Print the collected values (or perform any operation)
    print("Selected Features:", selected_features)
    print("Selected Classes:", selected_classes)
    print("Learning Rate:", learning_rate)
    print("Threshold:", threshold)
    print("Include Bias:", include_bias)
    print("Algorithm Type:", algorithm_type)









# create the main window #################################################################
mainWin = tk.Tk()
mainWin.title("Main Window")
mainWin.config(width=300, height=200)

intro = tk.Label(mainWin, text="Neural Network Task1")
intro.grid(row=0, column=0, columnspan=4, pady=10)  # Using grid instead of pack

frame = tk.Frame(mainWin)
frame.grid(padx=10, pady=10, row=1, column=0, columnspan=4)

# Features Frame
features_frame = tk.Frame(frame)
features_frame.grid(row=0, column=0, padx=20)
label_features = tk.Label(features_frame, text="Select 2 Features:")
label_features.grid(row=0, column=0, sticky='w')  # Using grid here instead of pack
features = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]

features_list = []
for i, feature in enumerate(features, 1):  # Added index to correctly place checkbuttons
    var = tk.IntVar()  # Create an IntVar for each option
    features_list.append(var)
    checkbutton = tk.Checkbutton(features_frame, text=feature, variable=var)
    checkbutton.grid(row=i, column=0, sticky='w')  # Using grid instead of pack

# Classes Frame
classes_frame = tk.Frame(frame)
classes_frame.grid(row=0, column=1, padx=20)
label_classes = tk.Label(classes_frame, text="Select 2 Classes:")
label_classes.grid(row=0, column=0, sticky='w')  # Using grid here instead of pack
classes = ["A", "B", "C"]

classes_list = []
for i, c in enumerate(classes, 1):  # Added index to correctly place checkbuttons
    var = tk.IntVar()  # Create an IntVar for each option
    classes_list.append(var)
    checkbutton = tk.Checkbutton(classes_frame, text=c, variable=var)
    checkbutton.grid(row=i, column=0, sticky='w')  # Using grid instead of pack

# Textboxes Frame
textboxes_frame = tk.Frame(frame)
textboxes_frame.grid(row=0, column=2, padx=20)

# Learning Rate
label_learning_rate = tk.Label(textboxes_frame, text="Learning Rate:")
label_learning_rate.grid(row=0, column=0, sticky='w')  # Using grid here instead of pack
learning_rate_entry = tk.Entry(textboxes_frame)
learning_rate_entry.grid(row=1, column=0, sticky='w')

# Threshold
label_threshold = tk.Label(textboxes_frame, text="Threshold:")
label_threshold.grid(row=2, column=0, sticky='w')  # Using grid here instead of pack
threshold_entry = tk.Entry(textboxes_frame)
threshold_entry.grid(row=3, column=0, sticky='w')

# Bias Checkbox
bias_var = tk.IntVar()
bias_checkbox = tk.Checkbutton(textboxes_frame, text="Include Bias", variable=bias_var)
bias_checkbox.grid(row=4, column=0, sticky='w', pady=(10, 10))

# Algorithms Frame
algorithms_frame = tk.Frame(frame)
algorithms_frame.grid(row=0, column=3, padx=20)

# Algorithm Type Radio Buttons
algorithm_var = tk.StringVar(value="Algorithm1")
label_algorithm = tk.Label(algorithms_frame, text="Algorithm Type:")
label_algorithm.grid(row=0, column=0, sticky='w')  # Using grid here instead of pack

radio_algorithm1 = tk.Radiobutton(algorithms_frame, text="Perceptron ", variable=algorithm_var, value="Algorithm1")
radio_algorithm2 = tk.Radiobutton(algorithms_frame, text="Adaline", variable=algorithm_var, value="Algorithm2")
radio_algorithm1.grid(row=1, column=0, sticky='w')  # Using grid instead of pack
radio_algorithm2.grid(row=2, column=0, sticky='w')  # Using grid instead of pack

# Submit Button
submit_button = tk.Button(frame, text="Submit", command=submit_action)
submit_button.grid(row=1, column=0, columnspan=4, pady=20)

mainWin.mainloop()

################################################