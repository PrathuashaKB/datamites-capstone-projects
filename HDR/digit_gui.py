import tkinter as tk
from tkinter import *
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Load your trained model
model = joblib.load("svm_model.joblib")

# GUI setup
window = Tk()
window.title("Handwritten Digit Recognition")
window.resizable(0, 0)

canvas_width, canvas_height = 200, 200
canvas = Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# PIL image to draw
image1 = Image.new("L", (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image1)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=15)
    draw.ellipse([x1, y1, x2, y2], fill='black')

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")
    label.config(text="Prediction: None")

def predict_digit():
    # Resize to 28x28 as in training
    img = image1.resize((28, 28))
    img = ImageOps.invert(img)  # Invert so background is black
    img = np.array(img)

    # Flatten for SVM input
    img = img.reshape(1, -1) / 255.0  

    # Predict
    prediction = model.predict(img)[0]
    label.config(text=f"Prediction: {prediction}")

# Buttons
btn_predict = Button(window, text="Predict", command=predict_digit)
btn_predict.grid(row=1, column=0, pady=2)

btn_clear = Button(window, text="Clear", command=clear_canvas)
btn_clear.grid(row=1, column=1, pady=2)

label = Label(window, text="Prediction: None", font=("Helvetica", 14))
label.grid(row=2, column=0, columnspan=2, pady=2)

canvas.bind("<B1-Motion>", paint)

window.mainloop()
