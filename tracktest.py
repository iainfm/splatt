#https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
import numpy as np
import cv2

# video capture object
video_capture = cv2.VideoCapture(1)

# Check the video stream started ok
assert video_capture.isOpened()

# Record session?
record = False
outfile = 'output.avi'


if record:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outfile,fourcc, 20.0, (640,480))

radius = 41 #must be an odd number, or else GaussianBlur will fail

# Plotting colours. TODO: vary over time
circleColor = (0, 0, 255)
circleThickness = 2
lineColor = (255, 255, 0)
lineThickness = 1

# List of captured points
storedTrace = []

while True:

    # Get the video frame
    ret, frame = video_capture.read()
    image = frame.copy()

    # Flip the image in both axes to get the position relative to the target correct
    image = cv2.flip(image, -1)

    # grey-ify the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    # Find the point of max brightness
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    
    # Add the discovered point to our list
    storedTrace.append(maxLoc)

    # Plot the traces so far
    for n in range(2, len(storedTrace)):

        cv2.circle(image, storedTrace[n], 5, circleColor, circleThickness)
        cv2.line(image, storedTrace[n-1], storedTrace[n], lineColor, lineThickness)

    # display the results
    cv2.imshow("Splatt", image)
   
    if record:
        # Write the frame to the output file
        out.write(image)

    # Check for user input
    keyPress = cv2.waitKey(1)

    if keyPress & 0xFF == ord('q'):
        # Quit
        video_capture.release()
        break
    else:
        if keyPress & 0xFF == ord('c'):
            # Clear the trace
            storedTrace = []