# Import necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import numpy as np
import random

# Function to augment image
def augment_image(image):
    # Rotate image by a random angle between -10 and 10 degrees
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-10, 10), 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Add Gaussian noise
    noise = np.random.normal(0, 0.05, image.shape) * 255
    noisy_image = cv2.add(rotated_image.astype(np.float32), noise.astype(np.float32)).astype(np.uint8)

    # Flip image horizontally with a probability of 0.5
    if random.random() > 0.5:
        flipped_image = cv2.flip(noisy_image, 1)
    else:
        flipped_image = noisy_image

    return flipped_image

# Path to dataset folder
dataset_path = "C:/Users/Dell/Pictures/Camera Roll/Dataset"

# List all image paths in the dataset
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images(dataset_path))

# Path to store encodings pickle file
encodingsP = "C:/Users/Dell/Pictures/Camera Roll/Dataset/encodings_new.pickle"

# Load existing encodings if pickle file exists
if os.path.exists(encodingsP):
    with open(encodingsP, 'rb') as f:
        data = pickle.load(f)
    knownEncodings = data['encodings']
    knownNames = data['names']
    print("[INFO] existing encodings loaded.")
else:
    knownEncodings = []
    knownNames = []
    print("[INFO] no existing encodings found. Creating new encodings.")

# Create a set of already processed image paths to avoid duplicates
processedImagePaths = set()

# Loop over the existing encodings to populate the set of processed image paths
for (encoding, name) in zip(knownEncodings, knownNames):
    # Recreate the image path based on the name and encoding
    person_dir = os.path.join(dataset_path, name)
    if os.path.exists(person_dir):  # Ensure the directory exists
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            processedImagePaths.add(img_path)

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Skip the image if it's already processed
    if imagePath in processedImagePaths:
        print(f"[INFO] skipping already processed image: {imagePath}")
        continue

    # Extract the person name from the image path
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    name = os.path.basename(os.path.dirname(imagePath))

    # Load the input image and convert it from BGR (OpenCV ordering) to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply augmentation and process the augmented images
    for _ in range(5):  # Create 5 augmented versions of each image
        augmented_image = augment_image(rgb)

        # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
        boxes = face_recognition.face_locations(augmented_image, model='cnn')

        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(augmented_image, boxes)

        # Loop over the encodings
        for encoding in encodings:
            # Add each encoding + name to our set of known names and encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

            # Add the newly processed image path to the set to avoid re-processing
            processedImagePaths.add(imagePath)

# Serialize the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {'encodings': knownEncodings, 'names': knownNames}
with open(encodingsP, 'wb') as f:
    f.write(pickle.dumps(data))
print("[INFO] encodings serialized to disk.")
