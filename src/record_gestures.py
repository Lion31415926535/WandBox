import cv2
import json
import os
import numpy as np

GESTURE_DIR = "D:/gestures"
os.makedirs(GESTURE_DIR, exist_ok=True)

positions = []
recording = False

def normalize(points, target_len=20):
    pts = np.array(points, dtype=float)
    pts -= pts[0]

    dist = np.linalg.norm(pts[-1] - pts[0])
    if dist > 0:
        pts /= dist

    idx = np.linspace(0, len(pts) - 1, target_len).astype(int)
    return pts[idx].tolist()

cap = cv2.VideoCapture(0)

print("Press R to record, S to save, Q to quit")

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
            center = (int(x), int(y))
            if recording:
                positions.append(center)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    cv2.imshow("Gesture Recorder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        positions.clear()
        recording = True
        print("Recording...")

    elif key == ord('s') and recording:
        name = input("Gesture name: ").strip()
        norm = normalize(positions)
        with open(f"{GESTURE_DIR}/{name}.json", "w") as f:
            json.dump(norm, f)
        print(f"Saved gesture '{name}'")
        recording = False

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
