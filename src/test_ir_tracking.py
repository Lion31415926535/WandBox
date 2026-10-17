# test_ir_tracking.py
import cv2

# Initialize the Video
cap = cv2.VideoCapture(0) # 0 specifies the camera being used (in this case it is the default camera)


while True:
    ret, frame = cap.read() # Take a frame capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale which is a prereq for thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY) # Change all the pixels in the image that are below a threshold to 0 

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds the outline of the pixels that are above the threshold

    center = None
    if contours:
        largest = max(contours, key=cv2.contourArea) # Finds the largest contour area
        if cv2.contourArea(largest) > 5:
            M = cv2.moments(largest) # Calculates image moments (see image moments Wikipedia article for more information)
            if M["m00"] > 0: 
                cx = int(M["m10"] / M["m00"]) # Calculates the x centroid of the image
                cy = int(M["m01"] / M["m00"]) # Calculates the y centroid of the image
                center = (cx, cy) # Creates a tuple of the image centroid
                cv2.circle(frame, center, 8, (0, 255, 0), -1) # Creates a circle on the image at the centroid of the brightest portion of the image

    cv2.imshow("IR Tracker", frame) # Shows the video
    if cv2.waitKey(1) == ord('q'): # Creates the method to stop the video capture 
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
