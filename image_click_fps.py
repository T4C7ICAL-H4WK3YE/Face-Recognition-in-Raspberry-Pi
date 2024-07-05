import cv2
import os

# Get the user's name
get_name = input('Enter name: ')
firstname = get_name.split()
name = firstname[0].lower()

# Create a directory with the name if it doesn't exist
if not os.path.exists(name):
    os.mkdir(name)
else:
    print(f"Folder '{name}' already exists or failed to create folder.")

# Change the current working directory to the created directory
os.chdir(name)

# Open the default camera (usually the first camera)
vid = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not vid.isOpened():
    print("Error: Could not open video device.")
    exit()

i = 0
while i < 200:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the resulting frame
    cv2.imshow('Camera Output', frame)
    
    # Save the captured frame
    cv2.imwrite(f'{name}{str(i)}.jpg', frame)
    
    i += 1
    
    # Press 'q' to exit the video window early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
