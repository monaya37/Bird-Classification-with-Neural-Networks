# App.py
import tkinter as tk
from Task1 import Task1


class App:
    def __init__(self, root):
        self.root = root

        self.root.title("ANN TASKS")
        self.root.geometry("600x600")
        self.root.configure(padx=20, pady=20)

        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=0, column=0, columnspan=2, sticky='ew')

        # Create buttons to switch between tabs using grid
        self.tab1_button = tk.Button(self.button_frame, text="Task 1", command=self.show_task1)
        self.tab1_button.grid(row=0, column=0, padx=(0, 0), pady=5)

        # Create frames for each tab
        self.task1_frame = Task1(self.root)  

        # Show Task 1 by default
        self.show_task1()  


    def show_task1(self):
        self.task1_frame.display()   

    
# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
