import os  # for accessing files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2  # for image processing
from PIL import Image #PIL -> pillow library used for image processing libraries
from sklearn.model_selection import train_test_split

# Create list containing files with masks
tumor = os.listdir('C:/Users/Khadeejas/Desktop/Brain tumor deployment project/Brain Tumor Data Set/Brain Tumor')
healthy = os.listdir('C:/Users/Khadeejas/Desktop/Brain tumor deployment project/Brain Tumor Data Set/Healthy')

tumor_labels = [0] * len(tumor)
healthy_labels = [1] * len(healthy)

labels = tumor_labels + healthy_labels 

def convert_images_to_numpy(image_dir, target_size=(128, 128)):
    data = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            with Image.open(img_path) as image:
                image = image.resize(target_size)  # resize image
                image = image.convert('RGB')       # convert to RGB
                image_array = np.array(image)      # convert to numpy array
                data.append(image_array)
    return data

# Paths to the directories containing images
tumor_path = 'C:/Users/Khadeejas/Desktop/Brain tumor deployment project/Brain Tumor Data Set/Brain Tumor'
healthy_path = 'C:/Users/Khadeejas/Desktop/Brain tumor deployment project/Brain Tumor Data Set/Healthy'

# Convert images in both directories to numpy arrays
data_tumor = convert_images_to_numpy(tumor_path)
data_healthy = convert_images_to_numpy(healthy_path)

# Combine the data from both directories
data = data_tumor + data_healthy

X = np.array(data)
Y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

x_train_scaled = x_train / 255.0  # to change the value from 0 to 1
x_test_scaled = x_test / 255.0

import tensorflow as tf
from tensorflow import keras

num_of_classes = 2

# Convert integer class labels to one-hot encoded vectors
Y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_of_classes)
Y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_of_classes)

input_shape = (128, 128, 3)

# Define and compile your model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_of_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_scaled, Y_train_one_hot, validation_split=0.1, epochs=10)

# Save the model
model.save('model.h5')
