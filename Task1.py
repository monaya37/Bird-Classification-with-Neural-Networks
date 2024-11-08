
import tkinter as tk
from tkinter import ttk, messagebox
from functions  import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, messagebox
from functions import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Task1:
    def __init__(self, parent):
        self.parent = parent
        self.functions_instance = functions(self)

        # Define larger font for the GUI components (Arial font)
        large_font = ('Helvetica', 12)  # Change the font to Arial

        # Create main frame
        self.frame = tk.Frame(self.parent)
        self.frame.grid(padx=10, pady=10)

        # Figure and canvas for plotting
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=4) 

        # Frame for Features
        self.features_frame = tk.Frame(self.frame)
        self.features_frame.grid(row=0, column=0, padx=20)

        self.label_features = tk.Label(self.features_frame, text="Select 2 Features:", font=large_font)
        self.label_features.pack(anchor='w')

        self.features = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]

        self.features_list = []
        for feature in self.features:
            var = tk.IntVar()  # Create an IntVar for each option
            self.features_list.append(var)
            checkbutton = tk.Checkbutton(self.features_frame, text=feature, variable=var, font=large_font)
            checkbutton.pack(anchor='w')

        # Frame for Classes
        self.classes_frame = tk.Frame(self.frame)
        self.classes_frame.grid(row=0, column=1, padx=20)

        self.label_classes = tk.Label(self.classes_frame, text="Select 2 Classes:", font=large_font)
        self.label_classes.pack(anchor='w')

        self.classes = ["A", "B", "C"]

        self.classes_list = []
        for c in self.classes:
            var = tk.IntVar()  # Create an IntVar for each option
            self.classes_list.append(var)
            checkbutton = tk.Checkbutton(self.classes_frame, text=c, variable=var, font=large_font)
            checkbutton.pack(anchor='w')

        # Frame for textboxes
        self.textboxes_frame = tk.Frame(self.frame)
        self.textboxes_frame.grid(row=0, column=2, padx=20)

        # Epochs
        self.label_epochs = tk.Label(self.textboxes_frame, text="Epochs:", font=large_font)
        self.label_epochs.pack(anchor='w')
        self.epochs_entry = tk.Entry(self.textboxes_frame, font=large_font)
        self.epochs_entry.pack(anchor='w')

        # Learning Rate
        self.label_learning_rate = tk.Label(self.textboxes_frame, text="Learning Rate:", font=large_font)
        self.label_learning_rate.pack(anchor='w')
        self.learning_rate_entry = tk.Entry(self.textboxes_frame, font=large_font)
        self.learning_rate_entry.pack(anchor='w')

        # Threshold
        self.label_threshold = tk.Label(self.textboxes_frame, text="Threshold:", font=large_font)
        self.label_threshold.pack(anchor='w')
        self.threshold_entry = tk.Entry(self.textboxes_frame, font=large_font)
        self.threshold_entry.pack(anchor='w')

        # Bias Checkbox
        self.bias_var = tk.IntVar()
        self.bias_checkbox = tk.Checkbutton(self.textboxes_frame, text="Include Bias", variable=self.bias_var, font=large_font)
        self.bias_checkbox.pack(anchor='w', pady=(10, 10))

        # Frame for algorithms
        self.algorithms_frame = tk.Frame(self.frame)
        self.algorithms_frame.grid(row=0, column=3, padx=20)

        # Algorithm Type Radio Buttons
        self.algorithm_var = tk.StringVar(value="Algorithm1")
        self.label_algorithm = tk.Label(self.algorithms_frame, text="Algorithm Type:", font=large_font)
        self.label_algorithm.pack(anchor='w')

        self.radio_algorithm1 = tk.Radiobutton(self.algorithms_frame, text="Perceptron", variable=self.algorithm_var, value="Algorithm1", font=large_font)
        self.radio_algorithm2 = tk.Radiobutton(self.algorithms_frame, text="Adaline", variable=self.algorithm_var, value="Algorithm2", font=large_font)
        self.radio_algorithm1.pack(anchor='w')
        self.radio_algorithm2.pack(anchor='w')

        # Button to show selected options
        self.submit_button = tk.Button(self.frame, text="Submit", command=lambda: self.functions_instance.run_algorithm(), font=large_font)
        self.submit_button.grid(row=1, column=0, columnspan=4, pady=20)

        self.dataset = None


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
