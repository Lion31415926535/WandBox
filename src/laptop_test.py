## Test the Gesture Matching ## 
import cv2
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Open the directory that stores the gestures
GESTURE_DIR = "/../gestures"
GESTURE_ACTIONS = {
    "open_box": "Open",
    "close_box": "Closed"}

THRESHOLD = 0.15
COOLDOWN = 3 # Cooldown between actions
last_trigger = 0

# Normalizes the centroid points
def normalize(points, target_len=20):
    pts = np.array(points, dtype=float) # Convert the positions to a numpy array
    pts -= pts[0] # Subtract the first position from all positions
    dist = np.linalg.norm(pts[-1] - pts[0])
    if dist > 0:
        pts /= dist
    idx = np.linspace(0, len(pts) - 1, target_len).astype(int)
    return pts[idx]

# Function to load the gestures
def load_gestures():
    gestures = {}
    for file in os.listdir(GESTURE_DIR):
        if file.endswith(".json"):
            with open(f"{GESTURE_DIR}/{file}") as f:
                gestures[file[:-5]] = np.array(json.load(f))

    print('Gestures Loaded')
    for name, content in gestures.items():
        print(f"Dictionary Name: {name}")
    
    return gestures

def gesture_distance(a, b):
    return np.mean(np.linalg.norm(a - b, axis=1))

# Function that controls actuators when an action is determined
def do_action(action):
    if action == "open":
        print("Opening box")
    elif action == "close":
        print("Closing box")

gestures = load_gestures()
positions = []

# Initialize the Video 
cap = cv2.VideoCapture(0) # 0 specifies the camera being used (in this case it is the default camera)

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
            positions.append(center) # Appends the brightess section centroid to a positions list
            if len(positions) > 30: 
                positions.pop(0) # Removes the first position if the length of the positions is longer than 30
            cv2.circle(frame, center, 5, (0, 255, 0), -1) # Creates a circle on the image at the centroid of the brightest portion of the image


    if len(positions) == 30:
        candidate = normalize(positions) # Normalize the centroid positions
        now = time.time() # Determine the current time
        if now - last_trigger > COOLDOWN: # Determine whether it has been long enough since the last action
            for name, template in gestures.items():
                score = gesture_distance(candidate, template)
                if score < THRESHOLD:
                    action = GESTURE_ACTIONS.get(name)
                    if action:
                        do_action(action)
                        last_trigger = now
                        positions.clear()
                        break

    cv2.imshow("Magic Wand", frame) # Shows the video
    if cv2.waitKey(1) == ord('q'): # Creates the method to stop the video capture
        break

    # Plot a user specified gesture
    elif cv2.waitKey(1) == ord('l'):
        name = input("Gesture name: ").strip()
        coordinates = gestures[name]
        x = coordinates[:,0]
        y = coordinates[:,1]

        plt.figure()
        plt.plot(x,y)
        plt.title(name)
        plt.show()

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
