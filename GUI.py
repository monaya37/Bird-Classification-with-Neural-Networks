# App.py
import tkinter as tk
from Task1 import Task1
from Task2 import Task2


class App:
    def __init__(self, root):
        self.root = root

        large_font = ('Helvetica', 14)  # Change the font to Arial

        self.root.title("ANN TASKS")
        self.root.geometry("1050x980")
        self.root.configure(padx=20, pady=20)

        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=0, column=0, columnspan=2, sticky='ew')

        # Create buttons to switch between tabs using grid
        self.tab1_button = tk.Button(self.button_frame, text="Task 1", command=self.show_task1, font=large_font)
        self.tab1_button.grid(row=0, column=0, padx=(0, 0), pady=5)

        # Create buttons to switch between tabs using grid
        self.tab2_button = tk.Button(self.button_frame, text="Task 2", command=self.show_task2, font=large_font)
        self.tab2_button.grid(row=0, column=1, padx=(0, 0), pady=5)

        # Create frames for each tab
        self.task1_frame = Task1(self.root)  
        self.task2_frame = Task2(self.root)  

        # Show Task 1 by default
        self.show_task1()  


    def show_task1(self):
        self.task2_frame.hide()
        self.task1_frame.display()  

    def show_task2(self):
        self.task1_frame.hide()
        self.task2_frame.display()   

    
# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
