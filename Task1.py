import tkinter as tk
from tkinter import ttk, messagebox

class Task1:
    def __init__(self, parent):
        self.parent = parent

        # Create main frame
        self.frame = tk.Frame(self.parent)
        self.frame.grid(padx=10, pady=10)

        # Frame for Features
        self.features_frame = tk.Frame(self.frame)
        self.features_frame.grid(row=0, column=0, padx=20)

        self.label_features = tk.Label(self.features_frame, text="Select Features:")
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

        self.label_classes = tk.Label(self.classes_frame, text="Select Classes:")
        self.label_classes.pack(anchor='w')

        self.classes = ["C1", "C2", "C3"]

        self.classes_list = []
        for c in self.classes:
            var = tk.IntVar()  # Create an IntVar for each option
            self.classes_list.append(var)
            checkbutton = tk.Checkbutton(self.classes_frame, text=c, variable=var)
            checkbutton.pack(anchor='w')

        # Frame for textboxes
        self.textboxes_frame = tk.Frame(self.frame)
        self.textboxes_frame.grid(row=0, column=2, padx=20)

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

        self.radio_algorithm1 = tk.Radiobutton(self.algorithms_frame, text="Algorithm 1", variable=self.algorithm_var, value="Algorithm1")
        self.radio_algorithm2 = tk.Radiobutton(self.algorithms_frame, text="Algorithm 2", variable=self.algorithm_var, value="Algorithm2")
        self.radio_algorithm1.pack(anchor='w')
        self.radio_algorithm2.pack(anchor='w')

        # Button to show selected options
        self.submit_button = tk.Button(self.frame, text="Submit", command=self.show_selected)
        self.submit_button.grid(row=1, column=0, columnspan=4, pady=20)



    def show_selected(self):
        # Show selected options using getter functions
        messagebox.showinfo("Selected Options", 
                            f"Features: {self.get_chosen_features()}\n"
                            f"Classes: {self.get_chosen_classes()}\n"
                            f"Learning Rate: {self.get_learning_rate()}\n"
                            f"Threshold: {self.get_threshold()}\n"
                            f"Include Bias: {self.bias_var.get()}\n"
                            f"Algorithm Type: {self.get_algorithm_type()}")

    def get_learning_rate(self):
        return self.learning_rate_entry.get()

    def get_threshold(self):
        return self.threshold_entry.get()

    def get_chosen_features(self):
        return [feature for feature, var in zip(self.features, self.features_list) if var.get()]

    def get_chosen_classes(self):
        return [c for c, var in zip(self.classes, self.classes_list) if var.get()]

    def get_algorithm_type(self):
        return self.algorithm_var.get()


    def display(self):
        self.frame.grid(row=1, column=0, columnspan=2, sticky='nsew')

    def hide(self):
        self.frame.grid_forget()