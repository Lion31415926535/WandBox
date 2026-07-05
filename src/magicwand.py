import cv2
import json
import os
import time
import numpy as np
import pigpio

# --------------------
# SERVO SETUP
# --------------------
SERVO_GPIO = 18
SERVO_OPEN = 2000
SERVO_CLOSED = 1000

pi = pigpio.pi()
pi.set_servo_pulsewidth(SERVO_GPIO, SERVO_CLOSED)

# --------------------
# GESTURES
# --------------------
GESTURE_DIR = "gestures"
GESTURE_ACTIONS = {
    "open_box": "open",
    "close_box": "close"
}

THRESHOLD = 0.15
COOLDOWN = 3
last_trigger = 0

def normalize(points, target_len=20):
    pts = np.array(points, dtype=float)
    pts -= pts[0]
    dist = np.linalg.norm(pts[-1] - pts[0])
    if dist > 0:
        pts /= dist
    idx = np.linspace(0, len(pts) - 1, target_len).astype(int)
    return pts[idx]

def load_gestures():
    gestures = {}
    for file in os.listdir(GESTURE_DIR):
        if file.endswith(".json"):
            with open(f"{GESTURE_DIR}/{file}") as f:
                gestures[file[:-5]] = np.array(json.load(f))
    return gestures

def gesture_distance(a, b):
    return np.mean(np.linalg.norm(a - b, axis=1))

def do_action(action):
    if action == "open":
        print("Opening box")
        pi.set_servo_pulsewidth(SERVO_GPIO, SERVO_OPEN)
    elif action == "close":
        print("Closing box")
        pi.set_servo_pulsewidth(SERVO_GPIO, SERVO_CLOSED)

gestures = load_gestures()
positions = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 10:
            (x, y), _ = cv2.minEnclosingCircle(largest)
            positions.append((int(x), int(y)))
            if len(positions) > 30:
                positions.pop(0)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    if len(positions) == 30:
        candidate = normalize(positions)
        now = time.time()
        if now - last_trigger > COOLDOWN:
            for name, template in gestures.items():
                score = gesture_distance(candidate, template)
                if score < THRESHOLD:
                    action = GESTURE_ACTIONS.get(name)
                    if action:
                        do_action(action)
                        last_trigger = now
                        positions.clear()
                        break

    cv2.imshow("Magic Wand", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pi.set_servo_pulsewidth(SERVO_GPIO, 0)
pi.stop()
