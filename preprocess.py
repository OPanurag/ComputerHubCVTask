# import cv2
# import numpy as np
#
#
# def preprocess_image(image_path):
#     """
#     1. Read the image
#     2. Image Enhancement
#             2.1 Gaussian Blur
#             2.2 Edge Detection using Canny
#     3. Return Edges
#     """
#
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#
#     return edges
#
#
# def preprocess_image_for_prediction(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image_resized = cv2.resize(image, (256, 256))
#     image_normalized = image_resized / 255.0
#     image_expanded = np.expand_dims(image_normalized, axis=0)
#     if image_expanded.shape[-1] != 1:
#         image_expanded = np.expand_dims(image_expanded, axis=-1)
#     return image_expanded
#
#


import cv2
import numpy as np


def preprocess_image(image_path):
    """
    1. Read the image
    2. Image Enhancement
            2.1 Gaussian Blur
            2.2 Edge Detection using Canny
    3. Return Edges
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    return edges


def count_edges(image_path):
    edges = preprocess_image(image_path)
    edge_count = np.sum(edges > 0)
    return edge_count
