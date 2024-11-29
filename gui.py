import tkinter as tk
from tkinter import messagebox
import csv
import subprocess

result = []

def submit_data():
    weight = weight_entry.get()
    hull_length = hull_length_entry.get()
    hull_width = hull_width_entry.get()
    hull_height = hull_height_entry.get()
    hole_length = hole_length_entry.get()
    hole_area = hole_area_entry.get()

    if not all([weight, hull_length, hull_width, hull_height, hole_length, hole_area]):
        messagebox.showwarning("Input Error", "All fields must be filled!")
        return

    try:
        weight = float(weight)
        hull_length = float(hull_length)
        hull_width = float(hull_width)
        hull_height = float(hull_height)
        hole_length = float(hole_length)
        hole_area = float(hole_area)

        result.append([weight, hull_length, hull_width, hull_height, hole_length, hole_area])
        
        save_data([weight, hull_length, hull_width, hull_height, hole_length, hole_area])  
        predict_latest() 

        weight_entry.delete(0, tk.END)
        hull_length_entry.delete(0, tk.END)
        hull_width_entry.delete(0, tk.END)
        hull_height_entry.delete(0, tk.END)
        hole_length_entry.delete(0, tk.END)
        hole_area_entry.delete(0, tk.END)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values!")

def save_data(new_entry):
    file_exists = True
    try:
        with open("data1.csv", "r") as file:
            pass
    except FileNotFoundError:
        file_exists = False

    with open("data1.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['weight', 'hull_length', 'hull_width', 'hull_height', 'hole_length', 'hole_area'])
        writer.writerow(new_entry)
    print("Data appended to data1.csv.")

def predict_latest():
    try:
        result = subprocess.run(["python", "test.py"], capture_output=True, text=True)
        prediction_output = result.stdout.strip()
        messagebox.showinfo("Prediction", f"Prediction:\n{prediction_output}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while running predictions: {e}")

root = tk.Tk()
root.title("Shipwreck Data Collection")
root.configure(bg="#87CEEB")
label_font = ("Arial", 12, "bold")
entry_bg = "#E0FFFF"
entry_fg = "#000080"

tk.Label(root, text="Weight of Ship (tons):", bg="#87CEEB", fg="white", font=label_font).grid(row=0, column=0, padx=10, pady=5)
weight_entry = tk.Entry(root, bg=entry_bg, fg=entry_fg)
weight_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Hull Length (meters):", bg="#87CEEB", fg="white", font=label_font).grid(row=1, column=0, padx=10, pady=5)
hull_length_entry = tk.Entry(root, bg=entry_bg, fg=entry_fg)
hull_length_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Hull Width (meters):", bg="#87CEEB", fg="white", font=label_font).grid(row=2, column=0, padx=10, pady=5)
hull_width_entry = tk.Entry(root, bg=entry_bg, fg=entry_fg)
hull_width_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Hull Height (meters):", bg="#87CEEB", fg="white", font=label_font).grid(row=3, column=0, padx=10, pady=5)
hull_height_entry = tk.Entry(root, bg=entry_bg, fg=entry_fg)
hull_height_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Hole Length (meters):", bg="#87CEEB", fg="white", font=label_font).grid(row=4, column=0, padx=10, pady=5)
hole_length_entry = tk.Entry(root, bg=entry_bg, fg=entry_fg)
hole_length_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Hole Area (square meters):", bg="#87CEEB", fg="white", font=label_font).grid(row=5, column=0, padx=10, pady=5)
hole_area_entry = tk.Entry(root, bg=entry_bg, fg=entry_fg)
hole_area_entry.grid(row=5, column=1, padx=10, pady=5)

submit_button = tk.Button(root, text="Submit", command=submit_data, bg="#4682B4", fg="white", font=("Arial", 10, "bold"))
submit_button.grid(row=6, column=0, columnspan=2, pady=10)

root.protocol("WM_DELETE_WINDOW", root.destroy)
root.mainloop()
