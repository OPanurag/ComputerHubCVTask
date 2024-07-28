import cv2
import numpy as np
import os


# Directory to save synthetic images
output_dir = 'synthetic_data'
os.makedirs(output_dir, exist_ok=True)


# Function to create a synthetic image of a stack of sheets
def create_sheet_stack_image(sheet_count):
    height, width = 256, 256
    image = np.ones((height, width), dtype=np.uint8) * 255  # White background

    # Draw sheets
    for i in range(sheet_count):
        y = height - (i + 1) * 5
        cv2.rectangle(image, (20, y), (width - 20, y + 5), (0, 0, 0), -1)

    return image


# Generate synthetic images and save them with annotations
annotations = []
for i in range(1, 51):  # Generating for 1 to 50 sheets
    image = create_sheet_stack_image(i)
    filename = f'sheet_stack_{i}.png'
    cv2.imwrite(os.path.join(output_dir, filename), image)
    annotations.append((filename, i))

# Save annotations to a CSV file
import csv
with open(os.path.join(output_dir, 'annotations.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'count'])
    writer.writerows(annotations)
