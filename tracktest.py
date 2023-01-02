# Splatt! Target shooting training system

# References:
#   https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
#   https://github.com/spatialaudio/python-sounddevice/issues/316

import numpy as np
import cv2
import sounddevice as sd # pip install sounddevice

# video capture object
video_capture_device = 0 # TODO: make this better
video_capture = cv2.VideoCapture(video_capture_device)

# Check the video stream started ok
assert video_capture.isOpened()

# Recording options
record = False # (Do not set to true here)
outfile = 'output.avi'

radius = 41 # must be an odd number, or else GaussianBlur will fail

# Plotting colours
initLineColour = (0, 0, 255, 0) # (Blue, Green, Red)
shotColor = (255, 0, 255) # Magenta
shotSize = 10
lineThickness = 2

# Tuple of line colours
lineColor = []

# Rates of colour change per frame (b, g, r)
dC = (0, 15, -15)

# List of captured points
storedTrace = []
startTrace = 1
frames = 0

# Coordinates of the 'fired' shot
recordedShotLoc = []

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

# Shot tracking
shotFired = False

# Start listening TODO: check/assert it started
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
        else:
            circleColor = (0, 0, 255)

        lineColor.append(initLineColour)
    
        # Add the discovered point to our list
        storedTrace.append(maxLoc)

    # Plot the line traces so far
    for n in range(startTrace, len(storedTrace)):
        thisLineColor = list(lineColor[n])

        if not shotFired:

            # Change the colour of the traces based on dC[]
            for c in range (0,3):    
                thisLineColor[c] = thisLineColor[c] + dC[c]
                if thisLineColor[c] > 255:
                    thisLineColor[c] = 255
                elif thisLineColor[c] < 0:
                    thisLineColor[c] = 0
            lineColor[n] = tuple(thisLineColor)

        # Draw a line from the previous point to this one
        cv2.line(image, storedTrace[n-1], storedTrace[n], lineColor[n], lineThickness)

    # Draw the shot circle if it's been taken
    if recordedShotLoc:
        cv2.circle(image, recordedShotLoc, shotSize, shotColor, -1)
    
    # display the results
    cv2.imshow("Splatt", image)

    # Write the frame to the output file
    if record:
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
        lineColor = []
        shotFired = False
    
    elif keyPress & 0xFF == ord('r'):
        if not record:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(outfile,fourcc, 20.0, (640,480))
            record = True