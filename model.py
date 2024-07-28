from tensorflow.keras.models import load_model
import numpy as np
import cv2


def load_trained_model():
    model = load_model('model.keras')
    return model


def preprocess_image_for_prediction(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    if image_expanded.shape[-1] != 1:
        image_expanded = np.expand_dims(image_expanded, axis=-1)
    return image_expanded


def predict_sheet_count(image_path, model):
    processed_image = preprocess_image_for_prediction(image_path)
    prediction = model.predict(processed_image)
    return int(prediction[0][0])
