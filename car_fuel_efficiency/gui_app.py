import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

linear_model = joblib.load("model/linear_regression_model.joblib")
knn_model = joblib.load("model/knn_model.joblib")
svr_model = joblib.load("model/svr_model.joblib")
scaler = joblib.load("model/scaler.joblib")


def origin_to_number(origin_name):
    mapping = {'American': 1, 'European': 2, 'Japanese': 3}
    return mapping.get(origin_name, 1)

def predict_mpg():
    try:
        cylinders = int(combo_cylinders.get())
        displacement = float(spin_displacement.get())
        horsepower = float(spin_horsepower.get())
        weight = float(entry_weight.get())
        acceleration = float(spin_acceleration.get())
        model_year = int(combo_model_year.get())
        origin_value = origin_to_number(combo_origin.get())

        X_input = np.array([[cylinders, displacement, horsepower, weight,
                             acceleration, model_year, origin_value]])
        X_scaled = scaler.transform(X_input)

        mpg_lin = linear_model.predict(X_scaled)[0]
        mpg_knn = knn_model.predict(X_scaled)[0]
        mpg_svr = svr_model.predict(X_scaled)[0]

        result_text.set(f"""🔴 Linear Regression: {mpg_lin:.2f} mpg
🔴 KNN Regressor: {mpg_knn:.2f} mpg
🔴 SVR (Optimized): {mpg_svr:.2f} mpg""")

    except Exception as e:
        messagebox.showerror("خطا در ورودی", f"ورودی نادرست یا ناقص است:\n\n{e}")

root = tk.Tk()
root.title("Auto MPG Prediction")
root.geometry("480x560")
root.configure(bg="#ffffff")

BUTANE_RED = "#B22222"

style = ttk.Style()
style.theme_use("default")
style.configure("TFrame", background="#ffffff")
style.configure("TLabel", background="#ffffff", foreground=BUTANE_RED, font=("Segoe UI", 10))
style.configure("TButton", background=BUTANE_RED, foreground="white", font=("Segoe UI", 10, "bold"))
style.map("TButton", background=[("active", "#8B1A1A")])
style.configure("TCombobox", arrowsize=14)

title_label = ttk.Label(root, text="🚘 Auto MPG Prediction Tool", font=("Segoe UI", 15, "bold"), foreground=BUTANE_RED)
title_label.pack(pady=10)

frame = ttk.Frame(root, padding=10)
frame.pack(fill=tk.X, padx=20, pady=5)

ttk.Label(frame, text="Cylinders:").grid(row=0, column=0, sticky="w", pady=4)
combo_cylinders = ttk.Combobox(frame, values=[3, 4, 5, 6, 8], width=18, state="readonly")
combo_cylinders.set(4)
combo_cylinders.grid(row=0, column=1, padx=5, pady=4)

ttk.Label(frame, text="Displacement:").grid(row=1, column=0, sticky="w", pady=4)
spin_displacement = tk.Spinbox(frame, from_=60, to=460, width=20, increment=1)
spin_displacement.grid(row=1, column=1, padx=5, pady=4)

ttk.Label(frame, text="Horsepower:").grid(row=2, column=0, sticky="w", pady=4)
spin_horsepower = tk.Spinbox(frame, from_=40, to=240, width=20, increment=1)
spin_horsepower.grid(row=2, column=1, padx=5, pady=4)

ttk.Label(frame, text="Weight:").grid(row=3, column=0, sticky="w", pady=4)
entry_weight = ttk.Entry(frame, width=22)
entry_weight.grid(row=3, column=1, padx=5, pady=4)

ttk.Label(frame, text="Acceleration:").grid(row=4, column=0, sticky="w", pady=4)
spin_acceleration = tk.Spinbox(frame, from_=7, to=25, width=20, increment=0.1)
spin_acceleration.grid(row=4, column=1, padx=5, pady=4)

ttk.Label(frame, text="Model Year:").grid(row=5, column=0, sticky="w", pady=4)
combo_model_year = ttk.Combobox(frame, values=list(range(70, 87)), width=18, state="readonly")
combo_model_year.set(76)
combo_model_year.grid(row=5, column=1, padx=5, pady=4)

ttk.Label(frame, text="Origin:").grid(row=6, column=0, sticky="w", pady=4)
combo_origin = ttk.Combobox(frame, values=["American", "European", "Japanese"], width=18, state="readonly")
combo_origin.set("American")
combo_origin.grid(row=6, column=1, padx=5, pady=4)

predict_button = ttk.Button(root, text="🔍 Predict MPG", command=predict_mpg)
predict_button.pack(pady=14)

result_text = tk.StringVar()
result_label = ttk.Label(root, textvariable=result_text, font=("Segoe UI", 11), justify="left", foreground=BUTANE_RED)
result_label.pack(pady=10)

root.mainloop()
