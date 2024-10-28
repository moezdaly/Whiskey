import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tensorflow.keras import regularizers
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, Rescaling
from keras.optimizers import Adam
from keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import  preprocess_input
from tensorflow.keras import layers, models, applications
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16


import cv2
import pandas as pd
import numpy as np

from IPython.display import display
import logging

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("##############################################")
print("##############################################")
print("Testing regulizers")
print("##############################################")
print("##############################################")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define the directory and image properties
data_dir = "./trainingsdaten"
img_height = 768
img_width = 1024
input_shape = (img_height, img_width, 3)
epochs = 15
# Load the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',  # Labels are generated from directory structure
    label_mode='int',  # Ensure labels are integers
    color_mode='rgb',
    validation_split=0.2,  # 20% of the images get used for validation
    subset="training",
    seed=123,  # For reproducing results
    image_size=(img_height, img_width),  # Using full resolution
    batch_size=2
)

# Load the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',  # Ensure labels are integers
    color_mode='rgb',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=2
)

test_dir = "./testdaten"
# Load the test data with file paths
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=1,
    shuffle=False  # Important to not shuffle test data
)


# InceptionV3 Simple Model
base_model_inceptionv3 = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
base_model_inceptionv3.trainable = False
preprocess_input_inceptionv3 = applications.inception_v3.preprocess_input

print('##############################')
print('this is my first model with inceptionv3 that got the best result of 0.89 with config 16,32,32,32 with 0.01 regul')
print('##############################')

# InceptionV3 Complex Model with L2 Regularization
inception_16_3x32_001regul = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Lambda(preprocess_input_inceptionv3),
    base_model_inceptionv3,

    # Convolutional layers with L2 regularization
    layers.Conv2D(16, 3, padding='same', activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),

    layers.Flatten(),

    # Dense layer with L2 regularization
    layers.Dense(128, activation='sigmoid',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(2)
])

# Compile the model
inception_16_3x32_001regul.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Train the model
print("Training Complex Model with InceptionV3 and Regularization:")
inception_16_3x32_001regul_history = inception_16_3x32_001regul.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

# Testing phase remains the same
print('Test the model I just trained')

# Initialize lists to store results
image_names = []
actual_labels = []
predicted_labels = []
predicted_probs = []

class_names = test_ds.class_names

# Iterate over the test dataset
for images, labels in test_ds:
    # Get image name
    image_name = test_ds.file_paths[len(image_names)]

    # Make predictions
    preds = inception_16_3x32_001regul.predict(images)
    pred_probs = tf.nn.softmax(preds, axis=-1)

    # Get predicted label
    predicted_label = np.argmax(pred_probs, axis=-1)

    # Store results
    image_names.append(image_name)
    actual_labels.append(class_names[labels[0]])  # labels is already a numpy array
    predicted_labels.append(class_names[predicted_label[0]])  # predicted_label is already a numpy array
    predicted_probs.append(pred_probs.numpy()[0].tolist())

# Calculate misclassification rate
misclassified_count = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted])
total_samples = len(image_names)
misclassification_rate = misclassified_count / total_samples

# Create DataFrame
results_df3 = pd.DataFrame({
    'Image Name': image_names,
    'Actual Label': actual_labels,
    'Predicted Label': predicted_labels,
    'Predicted Probability': predicted_probs
})

print(misclassified_count, misclassification_rate)
display(results_df3.to_string())

print('##############################')
print('this is my first model with inceptionv3 that got the best result of 0.89 with config 128_64_32_16')
print('##############################')

# InceptionV3 Complex Model
inception_128_64_32_16 = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Lambda(preprocess_input_inceptionv3),
    base_model_inceptionv3,
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(2)
])

inception_128_64_32_16.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

print("Training Complex Model with InceptionV3:")
inception_128_64_32_16_history = inception_128_64_32_16.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

print('Test the model I just trained')
# Initialize lists to store results
image_names = []
actual_labels = []
predicted_labels = []
predicted_probs = []

class_names = test_ds.class_names

# Iterate over the test dataset
for images, labels in test_ds:
    # Get image name
    image_name = test_ds.file_paths[len(image_names)]

    # Make predictions
    preds = inception_128_64_32_16.predict(images)
    pred_probs = tf.nn.softmax(preds, axis=-1)

    # Get predicted label
    predicted_label = np.argmax(pred_probs, axis=-1)

    # Store results
    image_names.append(image_name)
    actual_labels.append(class_names[labels[0]])  # labels is already a numpy array
    predicted_labels.append(class_names[predicted_label[0]])  # predicted_label is already a numpy array
    predicted_probs.append(pred_probs.numpy()[0].tolist())

# Calculate misclassification rate
misclassified_count = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted])
total_samples = len(image_names)
misclassification_rate = misclassified_count / total_samples

# Create DataFrame
results_df3 = pd.DataFrame({
    'Image Name': image_names,
    'Actual Label': actual_labels,
    'Predicted Label': predicted_labels,
    'Predicted Probability': predicted_probs
})

print(misclassified_count, misclassification_rate)
display(results_df3.to_string())

print('##############################')
print('this is my first model with inceptionv3 that got the best result of 0.89 with config 16,32,32,32 with no conv2d')
# print('ORIGNIAL MODEL')
print('##############################')

# InceptionV3 Complex Model
inception_16_3x32_no_conv = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Lambda(preprocess_input_inceptionv3),
    base_model_inceptionv3,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(2)
])

inception_16_3x32_no_conv.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

print("Training Complex Model with InceptionV3:")
inception_16_3x32_no_conv_history = inception_16_3x32_no_conv.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

print('Test the model I just trained')
# Initialize lists to store results
image_names = []
actual_labels = []
predicted_labels = []
predicted_probs = []

class_names = test_ds.class_names

# Iterate over the test dataset
for images, labels in test_ds:
    # Get image name
    image_name = test_ds.file_paths[len(image_names)]

    # Make predictions
    preds = inception_16_3x32_no_conv.predict(images)
    pred_probs = tf.nn.softmax(preds, axis=-1)

    # Get predicted label
    predicted_label = np.argmax(pred_probs, axis=-1)

    # Store results
    image_names.append(image_name)
    actual_labels.append(class_names[labels[0]])  # labels is already a numpy array
    predicted_labels.append(class_names[predicted_label[0]])  # predicted_label is already a numpy array
    predicted_probs.append(pred_probs.numpy()[0].tolist())

# Calculate misclassification rate
misclassified_count = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted])
total_samples = len(image_names)
misclassification_rate = misclassified_count / total_samples

# Create DataFrame
results_df3 = pd.DataFrame({
    'Image Name': image_names,
    'Actual Label': actual_labels,
    'Predicted Label': predicted_labels,
    'Predicted Probability': predicted_probs
})

print(misclassified_count, misclassification_rate)
display(results_df3.to_string())

print('##############################')
print('this is my first model with inceptionv3 that got the best result of 0.89 with config 16,32,32,32 with 1 conv2d')
# print('ORIGNIAL MODEL')
print('##############################')

# InceptionV3 Complex Model
inception_16_3x32_1_conv = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Lambda(preprocess_input_inceptionv3),
    base_model_inceptionv3,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(2)
])

inception_16_3x32_1_conv.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

print("Training Complex Model with InceptionV3:")
inception_16_3x32_1_conv_history = inception_16_3x32_1_conv.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

print('Test the model I just trained')
# Initialize lists to store results
image_names = []
actual_labels = []
predicted_labels = []
predicted_probs = []

class_names = test_ds.class_names

# Iterate over the test dataset
for images, labels in test_ds:
    # Get image name
    image_name = test_ds.file_paths[len(image_names)]

    # Make predictions
    preds = inception_16_3x32_1_conv.predict(images)
    pred_probs = tf.nn.softmax(preds, axis=-1)

    # Get predicted label
    predicted_label = np.argmax(pred_probs, axis=-1)

    # Store results
    image_names.append(image_name)
    actual_labels.append(class_names[labels[0]])  # labels is already a numpy array
    predicted_labels.append(class_names[predicted_label[0]])  # predicted_label is already a numpy array
    predicted_probs.append(pred_probs.numpy()[0].tolist())

# Calculate misclassification rate
misclassified_count = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted])
total_samples = len(image_names)
misclassification_rate = misclassified_count / total_samples

# Create DataFrame
results_df3 = pd.DataFrame({
    'Image Name': image_names,
    'Actual Label': actual_labels,
    'Predicted Label': predicted_labels,
    'Predicted Probability': predicted_probs
})

print(misclassified_count, misclassification_rate)
display(results_df3.to_string())

print('##############################')
print('this is my first model with inceptionv3 that got the best result of 0.89 with config 16,32,32,32 with 2 conv2d')
# print('ORIGNIAL MODEL')
print('##############################')

# InceptionV3 Complex Model
inception_16_3x32_2_conv = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Lambda(preprocess_input_inceptionv3),
    base_model_inceptionv3,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(2)
])

inception_16_3x32_2_conv.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

print("Training Complex Model with InceptionV3:")
inception_16_3x32_2_conv_history = inception_16_3x32_2_conv.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

print('Test the model I just trained')
# Initialize lists to store results
image_names = []
actual_labels = []
predicted_labels = []
predicted_probs = []

class_names = test_ds.class_names

# Iterate over the test dataset
for images, labels in test_ds:
    # Get image name
    image_name = test_ds.file_paths[len(image_names)]

    # Make predictions
    preds = inception_16_3x32_2_conv.predict(images)
    pred_probs = tf.nn.softmax(preds, axis=-1)

    # Get predicted label
    predicted_label = np.argmax(pred_probs, axis=-1)

    # Store results
    image_names.append(image_name)
    actual_labels.append(class_names[labels[0]])  # labels is already a numpy array
    predicted_labels.append(class_names[predicted_label[0]])  # predicted_label is already a numpy array
    predicted_probs.append(pred_probs.numpy()[0].tolist())

# Calculate misclassification rate
misclassified_count = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted])
total_samples = len(image_names)
misclassification_rate = misclassified_count / total_samples

# Create DataFrame
results_df3 = pd.DataFrame({
    'Image Name': image_names,
    'Actual Label': actual_labels,
    'Predicted Label': predicted_labels,
    'Predicted Probability': predicted_probs
})

print(misclassified_count, misclassification_rate)
display(results_df3.to_string())

# InceptionV3 Simple Model
base_model_inceptionv3 = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
base_model_inceptionv3.trainable = False
preprocess_input_inceptionv3 = applications.inception_v3.preprocess_input

print('##############################')
print('this is my first model with inceptionv3 that got the best result of 0.89 with config 16,32x4')
print('##############################')

# InceptionV3 Complex Model
inception_16_4x32 = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Lambda(preprocess_input_inceptionv3),
    base_model_inceptionv3,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(2)
])

inception_16_4x32.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

print("Training Complex Model with InceptionV3:")
inception_16_4x32_history = inception_16_4x32.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

print('Test the model I just trained')
# Initialize lists to store results
image_names = []
actual_labels = []
predicted_labels = []
predicted_probs = []

class_names = test_ds.class_names

# Iterate over the test dataset
for images, labels in test_ds:
    # Get image name
    image_name = test_ds.file_paths[len(image_names)]

    # Make predictions
    preds = inception_16_4x32.predict(images)
    pred_probs = tf.nn.softmax(preds, axis=-1)

    # Get predicted label
    predicted_label = np.argmax(pred_probs, axis=-1)

    # Store results
    image_names.append(image_name)
    actual_labels.append(class_names[labels[0]])  # labels is already a numpy array
    predicted_labels.append(class_names[predicted_label[0]])  # predicted_label is already a numpy array
    predicted_probs.append(pred_probs.numpy()[0].tolist())

# Calculate misclassification rate
misclassified_count = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual != predicted])
total_samples = len(image_names)
misclassification_rate = misclassified_count / total_samples

# Create DataFrame
results_df3 = pd.DataFrame({
    'Image Name': image_names,
    'Actual Label': actual_labels,
    'Predicted Label': predicted_labels,
    'Predicted Probability': predicted_probs
})

print(misclassified_count, misclassification_rate)
display(results_df3.to_string())