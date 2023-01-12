# Splatt! Target shooting training system

# References:
#   https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
#   https://github.com/spatialaudio/python-sounddevice/issues/316

#  TODO:
#  Export of recorded data to CSV
#  Scaling according to simulated distance
#  Configuration (device IDs etc) - setup, store and retrieval

import numpy as np       # pip install numpy / apt install python-numpy
import cv2               # pip install opencv-python / apt install python3-opencv
import sounddevice as sd # pip install sounddevice (requires libffi-dev)
import random

# Debug settings
debug_level =  0 # 0 (off), 1 (info), 2 (detailed)
debug_max = 2    # max debug level

# Virtual shooting range (TODO: implement scaling)
simulated_range_length = 25 # yards (doesn't matter as long
real_range_length = 4       # as unit are the same)
scale_factor = simulated_range_length / real_range_length

# Target dimensions
target_name = '25 yard prone'
target_diameter = 51.39 # mm
target_filename = '2010BM_89-18_to.png'

# video capture object
video_capture_device = 1 # TODO: make this better
video_capture = cv2.VideoCapture(video_capture_device)
captured_image_flip_needed = False # Whether the camera is mounted upside down

# Check the video stream started ok
assert video_capture.isOpened()

video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_size = (video_width, video_height)
video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))

if debug_level > 0:
    print(video_width , 'x' , video_height, ' @ ', video_fps, ' fps.')

# Recording options
record_video = False # (Do not set to true here)
video_output_file = 'output.avi'
composite_output_file = 'composite.png'

# Audio and video processing options
blur_radius = 11          # must be an odd number, or else GaussianBlur will fail. Lower is better for picking out point sources
min_detection_value = 50  # Trigger value to detect the reference point
click_threshold = 100     # audio level that triggers a 'shot'

# Plotting colours and options
init_line_colour = (0, 0, 255, 0) # (Blue, Green, Red)
shot_colour = (255, 0, 255)       # Magenta
shot_size = 20                    # TODO: scale?
line_thickness = 2

# target_filename = "2010BM_89-18_640x480.png"
colour_change_rate = (0, 15, -15) # Rates of colour change per frame (b, g, r)

# Tuple of line colours
line_colour = []

# List of captured points
stored_trace = []
start_trace = 1
shots_fired = -1 # The 0th is the calibration shot TODO: reset on keypress

# Coordinates of the 'fired' shot
recorded_shot_loc = []

# Calibration / scaling
calib_XY = (0, 0)

# Sound capture parameters
audio_chunk_size = 4096
if debug_level > 0:
    print(sd.query_devices()) # Choose device numbers from here. TODO: Get/save config

audio_stream = sd.Stream(
  device = (1, 4),
  samplerate = 44100,
  channels = 1,
  blocksize = audio_chunk_size)

# Shot tracking
shot_fired = False
first_shot = True

# Open target png  and create target_image and composite_image based on video frame size
target_file_image = cv2.imread(target_filename)

if ( target_file_image.shape[0] != target_file_image.shape[1]):
    # Target image should be square
    print('Warning: Target image is not square')

# Resize the target image to fit the video frame
target_file_image = cv2.resize(target_file_image, (video_height, video_height))

# Create a new blank target image
blank_target_image = np.zeros([video_height, video_width, 3], dtype = np.uint8)

# And colour it the same as the bottom-left (is it?) pixel of the target file
blank_target_image[:, :] = target_file_image[0:1, 0:1]

# Calculate the horizontal offset for centering the target image within the frame
target_offset = int(blank_target_image.shape[1]/2 - target_file_image.shape[1]/2)

# Copy the target file into the blank target image
blank_target_image[0:target_file_image.shape[1], target_offset:target_file_image.shape[0] + target_offset] = target_file_image

# Copy the blank target image to the two images we use for display
target_image = blank_target_image.copy()
composite_image = blank_target_image.copy()

# Start listening and check it started
audio_stream.start()
assert audio_stream.active

# Functions

def initialise_trace():
    # Clear the trace and reload target_image
    global start_trace, stored_trace, recorded_shot_loc, line_colour, shot_fired, target_image, composite_image
    start_trace = 1
    stored_trace = []
    recorded_shot_loc = []
    line_colour = []
    shot_fired = False
    target_image = cv2.imread(target_filename)
    target_image = blank_target_image.copy()

def calibrate_offset():
    # Calibrate offset to point source
    global calib_XY
    calib_XY = (int((video_width / 2) - max_loc[0]), int((video_height / 2) - max_loc[1]))
    if debug_level > 0:
        print(calib_XY)

while True:

    # Get the video frame
    video_ret, video_frame = video_capture.read()
    captured_image = video_frame.copy()

    # If set, flip the image
    if captured_image_flip_needed:
        captured_image = cv2.flip(captured_image, -1)
        #captured_image = cv2.flip(captured_image, 1)

    # grey-ify the image
    grey_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image then find the brightest region
    grey_image = cv2.GaussianBlur(grey_image, (blur_radius, blur_radius), 0)

    # Find the point of max brightness
    (min_brightness, max_brightness, min_loc, max_loc) = cv2.minMaxLoc(grey_image)

    if debug_level > 1:
        print(max_brightness, '@', max_loc)

    # If minimum brightness not met skip the rest of the loop
    if max_brightness > min_detection_value:

        # Check for click
        audio_data, audio_overflowed = audio_stream.read(audio_chunk_size)
        volume_norm = np.linalg.norm(audio_data)*10
        
        # Add the discovered point to our list with the initial line colour
        max_loc_x = int(( max_loc[0] + calib_XY[0] ) ) # TODO: Figure out the maths
        max_loc_y = int(( max_loc[1] + calib_XY[1] ) ) # TODO: Figure out the maths

        if max_loc_x < 0:
            max_loc_x = 0
        elif max_loc_x > video_width:
            max_loc_x = video_width

        if max_loc_y < 0:
            max_loc_y = 0
        elif max_loc_y > video_height:
            max_loc_y = video_height
            
        stored_trace.append((max_loc_x, max_loc_y))
        line_colour.append(init_line_colour)

    # Plot the line traces so far
    for n in range(start_trace, len(stored_trace)):
        this_line_colour = list(line_colour[n])

        if not shot_fired:

            if volume_norm >= click_threshold:
                
                recorded_shot_loc = (max_loc[0] + calib_XY[0], max_loc[1] + calib_XY[1])
                shot_fired = True
                shots_fired += 1

            # Change the colour of the traces based on colour_change_rate[]
            for c in range (0,3):
                this_line_colour[c] = this_line_colour[c] + colour_change_rate[c]
                if this_line_colour[c] > 255:
                    this_line_colour[c] = 255
                elif this_line_colour[c] < 0:
                    this_line_colour[c] = 0
            line_colour[n] = tuple(this_line_colour)

        # Draw a line from the previous point to this one
        cv2.line(target_image, stored_trace[n-1], stored_trace[n], line_colour[n], line_thickness)

    # Draw the shot circle if it's been taken
    if recorded_shot_loc:
        cv2.circle(target_image, recorded_shot_loc, shot_size, shot_colour, -1)
        if not first_shot:
            composite_colour = (random.randint(127, 255), random.randint(127, 255), random.randint(127, 255))
            cv2.circle(composite_image, recorded_shot_loc, shot_size, composite_colour, line_thickness)
            
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 2
            font_thickness = 2
            line_type = 2

            cv2.putText(composite_image, str(shots_fired), 
                (int(recorded_shot_loc[0] - (shot_size / 2)), int(recorded_shot_loc[1] + (shot_size / 2))),
                font, font_scale, composite_colour, font_thickness, line_type)

        # Remember to do anything else required with the recorded shot location here (eg csv output)
        recorded_shot_loc = ()

        if first_shot:
            # Calibrate the system and clear the results
            calibrate_offset()
            initialise_trace()
            first_shot = False
    
    # display the results
    cv2.imshow("Splatt - Live Trace", target_image)
    cv2.imshow("Splatt - Composite", composite_image)

    if debug_level > 0:
        cv2.imshow("Splatt - Grey", grey_image)

    # Write the frame to the output file
    if record_video:
        video_out.write(target_image)

    # Check for user input
    key_press = cv2.waitKey(1) & 0xFF
    
    if key_press == ord('q'):
        # Quit
        video_capture.release()
        audio_stream.close()
        # TODO: close video file if open as well?
        break

    elif key_press == ord('c'):
        # Clear the trace
        initialise_trace()
    
    elif key_press == ord('v'):
        # Record the output as a movie
        if not record_video:
            # Define the codec and create VideoWriter object
            four_cc = cv2.VideoWriter_fourcc(*'XVID')
            video_out = cv2.VideoWriter(video_output_file, four_cc, video_fps, video_size)
            record_video = True
            print('Recording started. Output file:', video_output_file)
    
    elif key_press == ord('d'):
        # Increase debug level
        debug_level += 1
        if debug_level > debug_max:
            debug_level = False
        print('Debug level:', int(debug_level))
    
    elif key_press == ord('r'):
        # Reset for recalibration
        first_shot = True

    elif key_press == ord('s'):
        # Save the composite image (and clear it?)
        cv2.imwrite(composite_output_file, composite_image)
    
    elif key_press == ord('f'):
        # Change the flip mode
        captured_image_flip_needed = not captured_image_flip_needed
        print('Flip required:', captured_image_flip_needed)
