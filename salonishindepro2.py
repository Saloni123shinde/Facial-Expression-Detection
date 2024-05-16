import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model('C:/Users/Lenovo/Downloads/emotion_detection_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect emotion from an image
def detect_emotion(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection App")
        self.root.geometry("600x400")
        
        self.title_label = tk.Label(root, text="Emotion Detection App", font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=10)
        
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image, bg="#ADD8E6", font=("Helvetica", 12))
        self.upload_button.pack(pady=20)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.display_image(file_path)
            emotion = detect_emotion(file_path)
            self.result_label.config(text=f"Detected Emotion: {emotion}", fg="blue")
            
    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
