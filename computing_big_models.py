import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import seaborn as sns
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Rescaling
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf

import cv2
import os
import pandas as pd
import numpy as np

# Define image sizes
img_height = 768
img_width = 1024
input_shape = (img_height, img_width, 3)
epochs = 18
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Define the base models with pretrained weights and custom input shape
base_models = {
    'VGG16': applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape),
    'ResNet50': applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape),
    'InceptionV3': applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape),
    'MobileNetV2': applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
    'EfficientNetB0': applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
}

# Freeze the pretrained weights
for base_model in base_models.values():
    base_model.trainable = False

# Define the preprocessing functions
preprocess_functions = {
    'VGG16': applications.vgg16.preprocess_input,
    'ResNet50': applications.resnet50.preprocess_input,
    'InceptionV3': applications.inception_v3.preprocess_input,
    'MobileNetV2': applications.mobilenet_v2.preprocess_input,
    'EfficientNetB0': applications.efficientnet.preprocess_input
}

# Function to build simple model
def build_simple_model(base_model, preprocess_input):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess_input),
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(2)  # Output layer with 2 units (assuming binary classification)
    ])
    return model

# Function to build complex model
def build_complex_model(base_model, preprocess_input):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess_input),
        base_model,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(2)  # Output layer with 2 units (assuming binary classification)
    ])
    return model

# Assuming train_ds and val_ds are predefined tf.data.Dataset objects
data_dir = "./trainingsdaten"

# Load datasets with specific image sizes
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=2
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=2
)

# Training and evaluating both simple and complex models with different base models
#for model_name, base_model in base_models.items():
#    # Simple model
#    simple_model = build_simple_model(base_model, preprocess_functions[model_name])
#    simple_model.compile(
#        optimizer='adam',
#        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#        metrics=['accuracy'],
#    )
#    print(f"Training Simple Model with {model_name}:")
#    simple_history = simple_model.fit(
#        train_ds,
#        validation_data=val_ds,
#        epochs=epochs,
#        verbose=1
#    )
#
#    # Complex model
#    complex_model = build_complex_model(base_model, preprocess_functions[model_name])
#    complex_model.compile(
#        optimizer='adam',
#        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#        metrics=['accuracy'],
#    )
#    print(f"Training Complex Model with {model_name}:")
#    complex_history = complex_model.fit(
#        train_ds,
#        validation_data=val_ds,
#        epochs=epochs,
#        verbose=1
#    )

def main():


    opt_base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    opt_base_model.trainable = False
    opt_preprocess_input = applications.inception_v3.preprocess_input
    three_layers_model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess_input),
        base_model,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(2)  # Output layer with 2 units (assuming binary classification)
    ])
    four_layers_model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess_input),
        base_model,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(2)  # Output layer with 2 units (assuming binary classification)
    ])
    five_layers_model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess_input),
        base_model,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(2)  # Output layer with 2 units (assuming binary classification)
    ])
    three_layers_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    print("Training Complex Model with 3 layers:")
    three_layers_history = three_layers_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    four_layers_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    print("Training Complex Model with 4 layers:")
    four_layers_history = four_layers_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    five_layers_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    print("Training Complex Model with 5 layers:")
    five_layers_history = five_layers_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )





if __name__ == "__main__":
    main()
