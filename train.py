import os
import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential, layers
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import ssl
# Bypass SSL Verification
ssl._create_default_https_context = ssl._create_unverified_context
# Directories for training and testing data
train_dir = "/Users/Desktop/train"
test_dir = "/Users/Desktop/test"
# Parameters
IMG_SIZE = 128 # Image size (128x128 pixels)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 15
# Data Augmentation for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
 rescale=1.0 / 255.0,
 rotation_range=30,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True,
 brightness_range=[0.8, 1.2],
 fill_mode='nearest'
)
# Data preprocessing for testing
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
# Load data
train_data = train_datagen.flow_from_directory(
 train_dir,
 target_size=(IMG_SIZE, IMG_SIZE),
 batch_size=BATCH_SIZE,
 class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
 test_dir,
 target_size=(IMG_SIZE, IMG_SIZE),
 batch_size=BATCH_SIZE,
 class_mode='categorical',
 shuffle=False
)
print(f"Classes in training data: {train_data.class_indices}")
print(f"Classes in testing data: {test_data.class_indices}")
# Use MobileNetV2 as the base model
base_model = tf.keras.applications.MobileNetV2(
 input_shape=(IMG_SIZE, IMG_SIZE, 3),
 include_top=False,
 weights='imagenet'
)
# Freeze most of the layers
for layer in base_model.layers[:-30]:
 layer.trainable = False
# Build the model
model = Sequential([
 base_model,
 layers.GlobalAveragePooling2D(),
 layers.BatchNormalization(),
 layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
 layers.Dropout(0.4),
 layers.BatchNormalization(),
 layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
 layers.Dropout(0.3),
 layers.Dense(NUM_CLASSES, activation='softmax')
])
# Compile the model with label smoothing
model.compile(
 optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
 loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
 metrics=['accuracy'] 
)
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
 initial_learning_rate=1e-4,
 decay_steps=10000,
 alpha=1e-6
)
# Train the model
history = model.fit(
 train_data,
 epochs=EPOCHS,
 validation_data=test_data,
 callbacks=[early_stopping]
)
# Save the model
model.save("human_action_model.h5")
# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
# Classification Report
y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(test_data.class_indices.keys())))
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(NUM_CLASSES)
plt.xticks(tick_marks, list(test_data.class_indices.keys()), rotation=45)
plt.yticks(tick_marks, list(test_data.class_indices.keys()))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
# Test with an example image
def predict_action(image_path, model, class_indices):
    image = cv2.imread(image_path) 
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    predicted_class = list(class_indices.keys())[np.argmax(prediction)]
    return predicted_class
example_image_path = "/Users/Desktop/test/calling/Image_10899.jpg"
predicted_action = predict_action(example_image_path, model, test_data.class_indices)
print(f"The predicted action is: {predicted_action}")