## Record Gestures ## 
import cv2
import json
import os
import numpy as np

# Create the directory that stores the gestures
GESTURE_DIR = "/../gestures"
os.makedirs(GESTURE_DIR, exist_ok=True)

positions = [] # List of the brightest point centroid positions
recording = False # Determines if we are in recording mode

# Normalizes the centroid points (how does the normalization work)
def normalize(points, target_len=20):
    pts = np.array(points, dtype=float) # Convert the positions to a numpy array
    pts -= pts[0] # Subtract the first position from all positions

    dist = np.linalg.norm(pts[-1] - pts[0]) # 
    if dist > 0:
        pts /= dist

    idx = np.linspace(0, len(pts) - 1, target_len).astype(int)
    return pts[idx].tolist()

# Initialize the Video
cap = cv2.VideoCapture(0) # 0 specifies the camera being used (in this case it is the default camera)

print("Press R to record, S to save, Q to quit")

while True:
    ret, frame = cap.read() # Take a frame capture
    if not ret: # If the frame capture doesn't work then break
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale which is a prereq for thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY) # Change all the pixels in the image that are below a threshold to 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds the outline of the pixels that are above the threshold

    if contours:
        largest = max(contours, key=cv2.contourArea) # Finds the largest contour area
        if cv2.contourArea(largest) > 10:
            M = cv2.moments(largest) # Calculates image moments (see image moments Wikipedia article for more information)
            if M["m00"] > 0: 
                cx = int(M["m10"] / M["m00"]) # Calculates the x centroid of the image
                cy = int(M["m01"] / M["m00"]) # Calculates the y centroid of the image
                center = (cx, cy) # Creates a tuple of the image centroid
            if recording:
                positions.append(center) # Appends the brightess section centroid to a positions list
            cv2.circle(frame, center, 5, (0, 255, 0), -1) # Creates a circle on the image at the centroid of the brightest portion of the image

    cv2.imshow("Gesture Recorder", frame) # Shows the video
    key = cv2.waitKey(1) & 0xFF # Stores the user input

    # Switches to the recording mode
    if key == ord('r'):
        positions.clear()
        recording = True
        print("Recording...")

    # Saves a gesture after recording
    elif key == ord('s') and recording:
        name = input("Gesture name: ").strip() # Name the gesture
        norm = normalize(positions) # Normalize the centroid positions
        with open(f"{GESTURE_DIR}/{name}.json", "w") as f: # Saves the gesture positions as a json file
            json.dump(norm, f)
        print(f"Saved gesture '{name}'")
        recording = False

    # Creates the method to stop the video capture
    elif key == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
