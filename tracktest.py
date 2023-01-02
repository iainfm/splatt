# Splatt! Target shooting training system

# References:
#   https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
#   https://github.com/spatialaudio/python-sounddevice/issues/316

import numpy as np
import cv2
import sounddevice as sd # pip install sounddevice

# video capture object
video_capture_device = 1 # TODO: make this better
video_capture = cv2.VideoCapture(video_capture_device)

# Check the video stream started ok
assert video_capture.isOpened()

# Record session?
record = False
outfile = 'output.avi'

radius = 41 # must be an odd number, or else GaussianBlur will fail

# Plotting colours. TODO: vary better over time
circleColor = (0, 0, 255)
circleThickness = 2
lineColor = []

# Line colour variance with time (frame) parameters
startRed = 0
startGreen = 255
startBlue = 0
# Rates of change / frame
dr = 1
dg = -1
db = 1

lineThickness = 1

# List of captured points
storedTrace = []
startTrace = 1
frames = 0
maxFrames = 255 # Max number of points/lines to plot
recordedShotLoc = []

# Initial line colours
red = startRed
green = startGreen
blue = startBlue

# Trigger value to detect the reference point
minDetectionValue = 50

# Sound capture parameters
CHUNK = 4096
stream = sd.Stream(
  device=("Microphone Array (Realtek High , MME", "Speaker/HP (Realtek High Defini, MME"),
  samplerate=44100,
  channels=2,
  blocksize=CHUNK)
clickThreshold = 100 # audio level that triggers a 'shot'

shotFired = False

# Start listening
stream.start()

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
    print(maxVal)

    # If minimum brightness not met skip the rest of the loop
    if maxVal > minDetectionValue:

        # Check for click
        indata, overflowed = stream.read(CHUNK)
        volume_norm = np.linalg.norm(indata)*10

        if volume_norm >= clickThreshold:
            recordedShotLoc = maxLoc
            shotFired = True
            dr = 0
            dr = 0
            db = 0
            red = 255
            green = 0
            blue = 0
        else:
            circleColor = (0, 0, 255)

        # Vary the line colour TODO: improve this
        thisColour = (blue, green, red, 0)
        lineColor.append(thisColour)

        red += dr
        green += dg
        blue += db

        if red >= 255 or red <= 0:
            dr = -1 * dr

        if green >= 255 or green <= 0:
            dg = -1 * dg

        if blue >= 255 or blue <= 0:
            db = -1 * db

        # Add the discovered point to our list
        storedTrace.append(maxLoc)

        # TODO: Decide if this is a requirement (or an option) - disappearing trace after n frames.
        # frames += 1
        # if frames > maxFrames:
            #startTrace = frames - maxFrames

        # add a one-off circle
        cv2.circle(image, maxLoc, 5, circleColor, circleThickness)


    # Plot the line traces so far
    for n in range(startTrace, len(storedTrace)):
        cv2.line(image, storedTrace[n-1], storedTrace[n], lineColor[n], lineThickness)

    if recordedShotLoc:
        cv2.circle(image, recordedShotLoc, 1, circleColor, circleThickness)
        cv2.circle(image, recordedShotLoc, 10, circleColor, circleThickness)
    
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
        stream.close()
        break

    elif keyPress & 0xFF == ord('c'):
        # Clear the trace
        startTrace = 1
        storedTrace = []
        recordedShotLoc = []
        # Reset the line colours TODO: tidy this up
        lineColor = []
        startRed = 0
        startGreen = 255
        startBlue = 0
        dr = 1
        dg = -1
        db = 1
        red = startRed
        green = startGreen
        blue = startBlue
    
    elif keyPress & 0xFF == ord('r'):
        if not record:
            # For testing purposes, define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(outfile,fourcc, 20.0, (640,480))
            record = True