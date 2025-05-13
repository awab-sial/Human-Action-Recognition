from collections import Counter
import os
import cv2
import numpy as np
import keras
import tensorflow as tf
from keras import models
import matplotlib.pyplot as plt
IMG_SIZE = 128
# Load the saved model
model = models.load_model("/Users/Desktop/human_action_model.h5") #Update this with the correct path to your saved model
# Directories for training and testing data
train_dir = "/Users/Desktop/train"
test_dir = "/Users/Desktop/test"
# Data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
# Load training data
train_data = train_datagen.flow_from_directory(
 train_dir,
 target_size=(IMG_SIZE, IMG_SIZE),
 batch_size=32,
 class_mode='sparse'
)
model.compile(
 loss="sparse_categorical_crossentropy",
 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
 metrics=["accuracy"] # Ensure a flat list of metric objects
) 
# Print class distribution in training data
print("Class distribution in training data:")
print(Counter(train_data.classes))
# Load testing data
test_data= test_datagen.flow_from_directory(
 test_dir,
 target_size=(IMG_SIZE, IMG_SIZE),
 batch_size=32,
 class_mode='sparse',
 shuffle=False # Ensure consistent evaluation order
)
# Evaluate the model on the test dataset
print
("\nEvaluating the model on test data...")
# Display model accuracy
test_loss, test_acc= model.evaluate(test_data)
print(f"Test Accuracy:{test_acc* 100:.2f}%")
