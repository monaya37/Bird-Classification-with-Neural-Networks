
import tkinter as tk
from tkinter import ttk, messagebox
from Task2Functions  import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#task2 gui components
class Task2:
    def __init__(self, parent):
        self.parent = parent
        self.functions = Task2Functions(self)

        large_font = ('Helvetica', 14)  # Change the font to Arial

        # Create main frame
        self.frame = tk.Frame(self.parent)
        self.frame.grid(padx=10, pady=10)

        
        # Figure and canvas for plotting
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))

        #self.fig.subplots_adjust(wspace=0.5)  # Adjust the space between subplots

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=4)


        # Frame for Features
        self.layers_frame = tk.Frame(self.frame)
        self.layers_frame.grid(row=0, column=0, padx=20)

        # Hidden layers
        self.num_hidden_layers = tk.Label(self.layers_frame, text="Number of Hidden Layers:", font=large_font)
        self.num_hidden_layers.pack(anchor='w')
        self.num_hidden_layers = tk.Entry(self.layers_frame, font=large_font)
        self.num_hidden_layers.insert(0, "3")
        self.num_hidden_layers.pack(anchor='w')

        self.hidden_layers = tk.Label(self.layers_frame, text="Enter Hidden Layers:", font=large_font)
        self.hidden_layers.pack(anchor='w')
        self.hidden_layers = tk.Entry(self.layers_frame, font=large_font)
        self.hidden_layers.insert(0, "3")
        self.hidden_layers.pack(anchor='w')

        # Frame for Classes
        self.neurons_frame = tk.Frame(self.frame)
        self.neurons_frame.grid(row=0, column=1, padx=20)

        self.num_of_neurons = tk.Label(self.neurons_frame, text="Number of Neurons:", font=large_font)
        self.num_of_neurons.pack(anchor='w')
        self.num_of_neurons = tk.Entry(self.neurons_frame, font=large_font)
        self.num_of_neurons.insert(0, "3")
        self.num_of_neurons.pack(anchor='w')

        self.neurons = tk.Label(self.neurons_frame, text="Enter Neurons:", font=large_font)
        self.neurons.pack(anchor='w')
        self.neurons = tk.Entry(self.neurons_frame, font=large_font)
        self.neurons.insert(0, "3")
        self.neurons.pack(anchor='w')

        # Frame for textboxes
        self.textboxes_frame = tk.Frame(self.frame)
        self.textboxes_frame.grid(row=0, column=2, padx=20)

        # Epochs
        self.label_epochs = tk.Label(self.textboxes_frame, text="Epochs:", font=large_font)
        self.label_epochs.pack(anchor='w')
        self.epochs_entry = tk.Entry(self.textboxes_frame, font=large_font)
        self.epochs_entry.insert(0, "100")  #Default 100
        self.epochs_entry.pack(anchor='w')

        # Learning Rate
        self.label_learning_rate = tk.Label(self.textboxes_frame, text="Learning Rate:", font=large_font)
        self.label_learning_rate.pack(anchor='w')
        self.learning_rate_entry = tk.Entry(self.textboxes_frame, font=large_font)
        self.learning_rate_entry.insert(0, "0.01")
        self.learning_rate_entry.pack(anchor='w')


        # Bias Checkbox
        self.bias_var = tk.IntVar(value=1)
        self.bias_checkbox = tk.Checkbutton(self.textboxes_frame, text="Include Bias", variable=self.bias_var, font=large_font)
        self.bias_checkbox.pack(anchor='w', pady=(10, 10))

        # Frame for algorithms
        self.algorithms_frame = tk.Frame(self.frame)
        self.algorithms_frame.grid(row=0, column=3, padx=20)

        # Algorithm Type Radio Buttons
        self.algorithm_var = tk.StringVar(value="Sigmoid")
        self.label_algorithm = tk.Label(self.algorithms_frame, text="Algorithm Type:", font=large_font)
        self.label_algorithm.pack(anchor='w')

        self.radio_algorithm1 = tk.Radiobutton(self.algorithms_frame, text="Sigmoid", variable=self.algorithm_var, value="Sigmoid", font=large_font)
        self.radio_algorithm2 = tk.Radiobutton(self.algorithms_frame, text="Tanh", variable=self.algorithm_var, value="Sigmoid", font=large_font)
        self.radio_algorithm1.pack(anchor='w')
        self.radio_algorithm2.pack(anchor='w')

        # Button to show selected options
        self.submit_button = tk.Button(self.frame, text="Submit", command=lambda: self.functions.run_algorithm(), font=large_font)
        self.submit_button.grid(row=1, column=0, columnspan=4, pady=20)

        # White box for displaying accuracy
        self.accuracy_label_frame = tk.Frame(self.frame)
        self.accuracy_label_frame.grid(row=5, column=0, columnspan=4, pady=(20, 10))

        self.accuracy_label = tk.Label(self.accuracy_label_frame, text="Accuracy: N/A", font=large_font, bg="white", relief="solid", width=20, height=2)
        self.accuracy_label.pack()
       
        self.dataset = None
        self.features = None
        self.classes = None


    def show_selected(self):
        print("Selected Options", 
                        f"Learning Rate: {self.get_learning_rate()}\n"
                        f"Include Bias: {self.bias_var.get()}\n"
                        f"Algorithm Type: {self.get_algorithm_type()}\n")
        self.get_hidden_layers()
        self.get_neurons()
        
    # Getters
    def get_epochs(self):
        return self.epochs_entry.get()

    def get_learning_rate(self):
        return self.learning_rate_entry.get()


    def get_num_of_hidden_layers(self):
        return [feature for feature, var in zip(self.features, self.features_list) if var.get()]

    def get_hidden_layers(self):
        neuron_values = self.hidden_layers.get().split(",")  # Split by comma
        neuron_values = [int(x) for x in neuron_values]  # Convert to integers
        print(neuron_values)  # Output the list
    
    def get_num_of_neurons(self):
        return [feature for feature, var in zip(self.features, self.features_list) if var.get()]

    def get_neurons(self):
            neuron_values = self.neurons.get().split(",")  # Split by comma
            neuron_values = [int(x) for x in neuron_values]  # Convert to integers
            print(neuron_values)  # Output the list

    def get_bias_state(self):
        return self.bias_var.get()

    def get_algorithm_type(self):
        return self.algorithm_var.get()

        #GUI Functions
    def display(self):
        self.frame.grid(row=1, column=0, columnspan=2, sticky='nsew')

    def hide(self):
        self.frame.grid_forget()