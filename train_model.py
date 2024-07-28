import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocess import count_edges

# Load the synthetic dataset
data_dir = 'synthetic_data'
annotations = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))

# Function to load images and their corresponding labels
def load_data(data_dir, annotations):
    images = []
    labels = []
    for _, row in annotations.iterrows():
        image_path = os.path.join(data_dir, row['filename'])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0  # Normalize
        images.append(image)
        labels.append(count_edges(image_path))  # Use edge count as the label
    images = np.array(images)
    labels = np.array(labels)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    return images, labels

# Load data
X, y = load_data(data_dir, annotations)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Output layer for regression (sheet count)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

model = build_model()

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Train the model
checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True)
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)
