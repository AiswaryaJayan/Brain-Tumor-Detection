from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
model = load_model('C:/Users/Khadeejas/Desktop/Brain tumor deployment project/model.h5')

def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image_array = np.array(image)
    image_scaled = image_array / 255.0
    image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])
    return image_reshaped

@app.route('/')
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image = preprocess_image(file)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        if predicted_class == 0:
            return render_template('result1.html')
        else:
            return render_template('result2.html')

if __name__ == '__main__':
    app.run(debug=True)
