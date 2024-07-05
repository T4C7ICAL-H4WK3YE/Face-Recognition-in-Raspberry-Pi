#! /usr/bin/python

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import os
import requests
import json
from collections import deque

# Correct platform plug-in
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Initialize 'currentnames' to store recognized names
currentnames = deque(maxlen=2)  # Store up to 2 names
last_face_time = time.time()

# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = r"C:\Users\Dell\Pictures\Camera Roll\Dataset\encodings_new.pickle"

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the video stream and allow the camera sensor to warm up
# Set the src to the following
# src = 0 : for the built-in single webcam, could be your laptop webcam
# src = 2 : I had to set it to 2 in order to use the USB webcam attached to my laptop
vs = VideoStream(src=0, framerate=10).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# Set a threshold for face recognition
threshold = 0.4

# API endpoint details
url = "http://10.2.201.147:8100/update"

# Format for displaying names on the digital display
display_format = {
    1: {"one": " WELCOMES {} TO IIITH"},
    2: {"one": " WELCOMES {} & {} TO IIITH"},
}

# Define headers with content type for JSON data
headers = {
    'Content-Type': 'application/json'
}

# Function to update the display via API
def update_display(name):
    if name == "":
        payload = {
            "one": "",
            "two": " WELCOMES",
            "three": "    YOU",
            "four": ""
        }
    else:
        # Split name into title and actual name parts (assuming a simple split by space)
        parts = name.split()
        if len(parts) >= 2:
            title = parts[0]
            actual_name = ' '.join(parts[1:]).upper()
        else:
            title = ""
            actual_name = name.upper()

        payload = {
            "one": " WELCOMES",
            "two": "    {}".format(title),
            "three": " {}".format(actual_name),
            "four": " TO IIITH"
        }

    payload_json = json.dumps(payload)
    try:
        response = requests.post(url, headers=headers, data=payload_json)
        if response.status_code == 200:
            print("Display updated successfully:", payload)
        else:
            print("Failed to update display. Status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error updating display:", e)

    time.sleep(2)

# Loop over frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    # to 500px (to speed up processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    
    # Detect the face boxes
    boxes = face_recognition.face_locations(frame)
    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Calculate the distance between the input face encoding and the known face encodings
        distances = face_recognition.face_distance(data["encodings"], encoding)
        min_distance = min(distances) if distances.size > 0 else 1.0

        # Set the name to "Unknown" if the minimum distance is above the threshold
        name = "Unknown"
        if min_distance < threshold:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            if True in matches:
                # Find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # Loop over the matched indexes and maintain a count for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Determine the recognized face with the largest number of votes (note: in the event of an unlikely tie Python will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                # Check if the name is not already in currentnames and add it
                if name not in currentnames:
                    currentnames.append(name)
                    last_face_time = time.time()
                    print("Recognized:", name)
                    
                    # Update display after 2 seconds if no new face is detected
                    if len(currentnames) <= 2:
                        time.sleep(2)
                        update_display(name)

        # Update the list of names
        names.append(name)

    # Update display if no new face detected for 10 seconds
    if time.time() - last_face_time > 10:
        currentnames.clear()
        update_display("")

    # Update the FPS counter
    fps.update()

    # Loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    # Display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit when 'q' key is pressed
    if key == ord("q"):
        break

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
