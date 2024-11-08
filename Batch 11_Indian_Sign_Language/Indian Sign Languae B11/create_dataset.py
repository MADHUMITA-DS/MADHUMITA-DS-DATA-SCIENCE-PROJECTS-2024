import os
import pickle
import cv2
import mediapipe as mp
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # This will suppress info and warnings


# Import MediaPipe's hands module for hand detection and landmark estimation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

data_dir = './ISL dataset'
dataset = []
labels = []

# Loop through each directory (representing each class) inside the dataset folder
for directory in os.listdir(data_dir):
    path = os.path.join(data_dir, directory)  # Construct the full path for the current class directory

    # Loop through each image file in the current class directory
    for img_path in os.listdir(path):
        normalized_landmarks = []  # List to store normalized x, y coordinates
        x_coordinates, y_coordinates = [], []  # Temporary lists for x and y coordinates

        # Read the image
        image_path = os.path.join(path, img_path)
        image = cv2.imread(image_path)

        # Check if the image was successfully loaded
        if image is None:
            print(f"Warning: Unable to load image at path: {image_path}")
            continue  # Skip to the next image if loading failed

# Assuming you have the rest of the code as before:
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
processed_image = hands.process(image_rgb)

if processed_image.multi_hand_landmarks:
    for hand_landmark in processed_image.multi_hand_landmarks:
        x_coordinates, y_coordinates = [], []

        # Get coordinates of the hand landmarks
        for landmark in hand_landmark.landmark:
            x_coordinates.append(landmark.x)
            y_coordinates.append(landmark.y)

        # Normalize the landmarks manually (since image is square with 128x128)
        normalized_landmarks = []
        width, height = 128, 128
        for x, y in zip(x_coordinates, y_coordinates):
            normalized_x = x / width
            normalized_y = y / height
            normalized_landmarks.extend((normalized_x, normalized_y))

        # Append normalized landmarks to dataset
        dataset.append(normalized_landmarks)
        labels.append(directory)


# Save the dataset and labels using pickle
with open("./ISL.pickle", "wb") as f:
    pickle.dump({"dataset": dataset, "labels": labels}, f)