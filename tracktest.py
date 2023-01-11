# Splatt! Target shooting training system

# References:
#   https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
#   https://github.com/spatialaudio/python-sounddevice/issues/316

#  TODO:
#  Export of recorded data to CSV
#  Auto-calibration
#  Scaling according to simulated distance
#  Configuration (device IDs etc) - setup, store and retrieval

import numpy as np       # pip install numpy / apt install python-numpy
import cv2               # pip install opencv-python / apt install python3-opencv
import sounddevice as sd # pip install sounddevice (requires libffi-dev)

# Debug settings
debug = False # 0 (off), 1 (info), 2 (detailed)
debug_max = 2 # max debug level

# video capture object
video_capture_device = 1 # TODO: make this better
video_capture = cv2.VideoCapture(video_capture_device)

# Check the video stream started ok
assert video_capture.isOpened()

video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_size = (video_width, video_height)
video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))

if debug > 0:
    print(video_width , 'x' , video_height, ' @ ', video_fps, ' fps.')

# Recording options
record = False # (Do not set to true here)
outfile = 'output.avi'

# Audio and video processing options
radius = 11             # must be an odd number, or else GaussianBlur will fail. Lower is better for picking out point sources
minDetectionValue = 50  # Trigger value to detect the reference point
clickThreshold = 100    # audio level that triggers a 'shot'

# Plotting colours and options
initLineColour = (0, 0, 255, 0) # (Blue, Green, Red)
shotColour = (255, 0, 255) # Magenta
shotSize = 10 # TODO: scale?
lineThickness = 2
card_colour = (147, 182, 213) # Future use
targetFilename = "2010BM_89-18_640x480.png"

# Tuple of line colours
lineColour = []

# Rates of colour change per frame (b, g, r)
dC = (0, 15, -15)

# List of captured points
storedTrace = []
startTrace = 1
frames = 0

# Coordinates of the 'fired' shot
recordedShotLoc = []

# Calibration / scaling
calib_XY = (0, 0)

# Sound capture parameters
CHUNK = 4096
if debug > 0:
    print(sd.query_devices()) # Choose device numbers from here. TODO: Get/save config

stream = sd.Stream(
  device=(1, 4),
  samplerate=44100,
  channels=1,
  blocksize=CHUNK)

# Shot tracking
shotFired = False

# Open target png
target = cv2.imread(targetFilename)

# Start listening and check it started
stream.start()
assert stream.active

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

    if debug > 1:
        print(maxVal, '@', maxLoc)

    # If minimum brightness not met skip the rest of the loop
    if maxVal > minDetectionValue:

        # Check for click
        indata, overflowed = stream.read(CHUNK)
        volume_norm = np.linalg.norm(indata)*10
        
        # Add the discovered point to our list with the initial line colour
        storedTrace.append((maxLoc[0] + calib_XY[0], maxLoc[1] + calib_XY[1]))
        lineColour.append(initLineColour)

    # Plot the line traces so far
    for n in range(startTrace, len(storedTrace)):
        thisLineColour = list(lineColour[n])

        if not shotFired:

            if volume_norm >= clickThreshold:
                recordedShotLoc = (maxLoc[0] + calib_XY[0], maxLoc[1] + calib_XY[1])
                shotFired = True

            # Change the colour of the traces based on dC[]
            for c in range (0,3):
                thisLineColour[c] = thisLineColour[c] + dC[c]
                if thisLineColour[c] > 255:
                    thisLineColour[c] = 255
                elif thisLineColour[c] < 0:
                    thisLineColour[c] = 0
            lineColour[n] = tuple(thisLineColour)

        # Draw a line from the previous point to this one
        cv2.line(target, storedTrace[n-1], storedTrace[n], lineColour[n], lineThickness)

    # Draw the shot circle if it's been taken
    if recordedShotLoc:
        cv2.circle(target, recordedShotLoc, shotSize, shotColour, -1)
    
    # display the results
    cv2.imshow("Splatt", target)
    if debug > 0:
        cv2.imshow("Splatt - Grey", gray)

    # Write the frame to the output file
    if record:
        out.write(target)

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
        lineColour = []
        shotFired = False
        target = cv2.imread(targetFilename)
    
    elif keyPress & 0xFF == ord('r'):
        # Record the output as a movie
        if not record:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(outfile, fourcc, video_fps, video_size)
            record = True
    
    elif keyPress & 0xFF == ord('d'):
        # Increase debug level
        debug += 1
        if debug > debug_max:
            debug = False
        print('Debug level:', int(debug))

    elif keyPress & 0xFF == ord('k'):
        # Calibrate offset to point source
        # TODO: do this on the first shot? Clear trace after calibration?
        calib_XY = (int((video_width / 2) - maxLoc[0]), int((video_height / 2) - maxLoc[1]))
        if debug > 0:
            print(calib_XY)