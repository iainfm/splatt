# Splatt! Target shooting training system

# References:
#   https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
#   https://github.com/spatialaudio/python-sounddevice/issues/316
#   https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

#  TODO:
#  Export of recorded data to CSV
#  Configuration (device IDs etc) - setup, store and retrieval

print('Splatt initialising...')

import numpy as np       # pip install numpy / apt install python-numpy
import cv2               # pip install opencv-python / apt install python3-opencv
import sounddevice as sd # pip install sounddevice (requires libffi-dev)
import random
from time import time
from config import *     # read static variables etc
from splatt_functions import *

# Functions TODO: move to separate file

def initialise_trace(clear_composite: bool):
    # Clear the trace and reload target_image
    global stored_trace, recorded_shot_loc, line_colour, shot_fired, target_image, composite_image, shots_fired, auto_reset_time_expired, this_shot_time, shots_fired, calibration_shots_req, calibrated
    stored_trace = []
    recorded_shot_loc = ()
    line_colour = []
    shot_fired = False
    target_image = blank_target_image.copy()
    if clear_composite:
        composite_image = blank_target_image.copy()
        calibrated = False
    auto_reset_time_expired = False
    this_shot_time = time()

    if shots_fired >= calibration_shots_req:
        calibrated = True

# video capture object
video_capture = cv2.VideoCapture(video_capture_device, cv2.CAP_DSHOW)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check the video stream started ok
assert video_capture.isOpened()

video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_size = (video_width, video_height)
video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))

scaled_shot_radius = convert_to_pixels(0.5 * shot_calibre, target_diameter, video_height)

# Font options
font = cv2.FONT_HERSHEY_PLAIN
font_scale = np.ceil( scaled_shot_radius / 13 )

print(video_width , 'x' , video_height, ' @ ', video_fps, ' fps.') if debug_level > 0 else None
print(sd.query_devices()) if debug_level > 0 else None # Choose device numbers from here. TODO: Get/save config / -l(ist) option

audio_stream = sd.Stream(
  device = None,
  samplerate = 44100,
  channels = 1,
  blocksize = audio_chunk_size)

# Shot tracking
shot_fired = False
shots_fired = 0
# first_shot = True

# Open target png  and create target_image and composite_image based on video frame size
target_file_image = cv2.imread(target_filename)

# Target image should be square for scaling etc to work properly
print('Warning: Target image is not square') if ( target_file_image.shape[0] != target_file_image.shape[1]) else None

# Resize the target image to fit the video frame
target_file_image = cv2.resize(target_file_image, (video_height, video_height))

# Create a new blank target image
blank_target_image = np.zeros([video_height, video_width, 3], dtype = np.uint8)

# And colour it the same as the bottom-left (is it?) pixel of the target file
blank_target_image[:, :] = target_file_image[1:2, 1:2]

# Calculate the horizontal offset for centering the target image within the frame
target_offset = int(blank_target_image.shape[1]/2 - target_file_image.shape[1]/2)

# Copy the target file into the blank target image
blank_target_image[0:target_file_image.shape[1], target_offset:target_file_image.shape[0] + target_offset] = target_file_image

# Copy the blank target image to the two images we use for display
# target_image = blank_target_image.copy()
# composite_image = blank_target_image.copy()

# Start listening and check it started
audio_stream.start()
assert audio_stream.active

initialise_trace(True)

def draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image):
    shots_plotted = 0
    for recorded_shot in composite_shots:
        shots_plotted += 1
        composite_colour = (random.randint(1, 64) * 4 - 1, random.randint(32, 64) * 4 - 1, random.randint(1, 64) * 4 - 1)
        cv2.circle(composite_image, recorded_shot, scaled_shot_radius, composite_colour, line_thickness)
                
                # Number the shot on the composite image
        text_size = cv2.getTextSize(str(shots_plotted), font, font_scale, font_thickness)[0]
        text_X = int((recorded_shot[0] - (text_size[0]) / 2))
        text_Y = int((recorded_shot[1] + (text_size[1]) / 2))

        cv2.putText(composite_image, str(shots_plotted), 
                    (text_X, text_Y),
                    font, font_scale, composite_colour, font_thickness, line_type)

def draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image):
    # Find the bounding circle of all shots so far
    if len(composite_shots) > 1:
        (bc_X, bc_Y), bc_radius = cv2.minEnclosingCircle(np.asarray(composite_shots))
        bc_centre = (int(bc_X), int(bc_Y))
        cv2.circle(composite_image, bc_centre, int(bc_radius + scaled_shot_radius), (0, 255, 255), 2)
        actual_spread = convert_to_real(2 * bc_radius, target_diameter, video_height) # (mm)
        cv2.putText(composite_image, 'Spread: ' + str(np.around(actual_spread, 2)) + 'mm', (5, 25), font, 1, (0, 0, 0), 1, 1)


################################################## Main Loop ##################################################

while True:

    # Get the video frame
    video_ret, video_frame = video_capture.read()
    captured_image = video_frame.copy()

    # If set, flip the image
    if captured_image_flip_needed:
        captured_image = cv2.flip(captured_image, -1)

    # grey-ify the image and apply a Gaussian blur to the image
    grey_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.GaussianBlur(grey_image, (blur_radius, blur_radius), 0)

    # Find the point of max brightness
    (min_brightness, max_brightness, min_loc, max_loc) = cv2.minMaxLoc(grey_image)

    print(max_brightness, '@', max_loc) if ( target_file_image.shape[0] != target_file_image.shape[1]) else None

    # If minimum brightness not met skip the rest of the loop
    if max_brightness > detection_threshold:
        # TODO: On-screen indicator of whether the detection threshold is met
        
        # Check for click
        audio_data, audio_overflowed = audio_stream.read(audio_chunk_size)
        volume_norm = np.linalg.norm(audio_data[:, 0]) * 10
        
        # Offset the location based on the calibration
        max_loc_x = int(( max_loc[0] + calib_XY[0] ) )
        max_loc_y = int(( max_loc[1] + calib_XY[1] ) )
        
        if shots_fired >= calibration_shots_req:
            # Scale based on the virtual / real range distances, limited to video frame dimensions
            max_loc_x = np.clip(int( scale_factor * (max_loc_x - video_width / 2) + video_width /2 ), 0, video_width)
            max_loc_y = np.clip(int( scale_factor * (max_loc_y - video_height / 2) + video_height /2 ), 0, video_height)

            # Store the scaled coordinates (unless it's the calibration 'shot')
            max_loc = (max_loc_x, max_loc_y)
        
        # Add the discovered point to our list with the initial line colour
        stored_trace.append((max_loc_x, max_loc_y))
        line_colour.append(init_line_colour)

    # Append the new frame
    video_frames.append(target_image.copy()) if record_video else None

    # Plot the line traces so far
    for n in range(1, len(stored_trace)):
        this_line_colour = list(line_colour[n])

        if not shot_fired:

            # Check audio levels TODO: FFT analysis
            if volume_norm >= click_threshold:
                recorded_shot_loc = max_loc
                shot_fired = True
                shots_fired += 1
                shot_time = time() + auto_reset_time

            # Change the colour of the traces based on colour_change_rate[]
            for c in range (0,3):
                this_line_colour[c] = this_line_colour[c] + colour_change_rate[c]
                if this_line_colour[c] > 255:
                    this_line_colour[c] = 255
                elif this_line_colour[c] < 0:
                    this_line_colour[c] = 0
            line_colour[n] = tuple(this_line_colour)
        else:
            auto_reset_time_expired = True if time() > shot_time else False

        # Draw a line from the previous point to this one
        cv2.line(target_image, stored_trace[n-1], stored_trace[n], line_colour[n], line_thickness)

    # Draw the shot circle if the shot has been taken TODO: Check whether it looks better to plot this under the trace

    if recorded_shot_loc:

        cv2.circle(target_image, recorded_shot_loc, scaled_shot_radius, shot_colour, -1)

        if shots_fired > calibration_shots_req:

            composite_shots.append(recorded_shot_loc)

            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)

        else:
            calibration_shots.append(recorded_shot_loc)
            if debug_level > 0:
                # Mark the calibration shot position
                # TODO: Make this the default? Would need to unplot on undo
                cv2.circle(composite_image, recorded_shot_loc, 1, shot_colour, -1)
            
        # Remember to do anything else required with the recorded shot location here (eg csv output)
        recorded_shot_loc = ()

        if shots_fired == calibration_shots_req:
            # Calibrate the system and clear the results
            
            # Find the spread of calibration shots
            (cal_X, cal_Y), cal_radius = cv2.minEnclosingCircle(np.asarray(calibration_shots))
            calib_XY = calibrate_offset(video_width, video_height, cal_X, cal_Y)

            # Plot a circle to show the calibration data
            print('Calibration offset (px):', calib_XY) if debug_level >= 0 else None
            cv2.circle(target_image, ((int(cal_X), int(cal_Y))), int(cal_radius), (0, 255, 255), 2)
            cv2.circle(target_image, ((int(cal_X), int(cal_Y))), 2, (0, 255, 255), 2)

    if shots_fired < calibration_shots_req:
        cv2.putText(target_image, 'CALIBRATING', (5, 25), font, 2, (0, 0, calib_text_red), 1, 1)
        
        # Fancy calibration font colout
        calib_text_red += d_calib_text_red
        if calib_text_red > 255:
            calib_text_red = 255
            d_calib_text_red = -8
        elif calib_text_red < calib_text_red_min:
            calib_text_red = calib_text_red_min
            d_calib_text_red = 8

    else:
        if calibrated and display_shot_time:
            text = "%04.1f" % (time() - this_shot_time, )
            text_size = cv2.getTextSize(text, font, 2, font_thickness)[0]
            cv2.rectangle(target_image, (3, 23), (7 + text_size[0], 27 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(target_image, text, (5, 26 + text_size[1]), font, 2, (255, 255, 255), 1, 1, False)

    cv2.imshow('Splatt - Live trace', target_image)
    cv2.imshow('Splatt - Composite', composite_image)
    cv2.imshow('Splatt - Blurred Vision', grey_image) if debug_level > 0 else None

    if auto_reset_time_expired:
        if (shots_fired <= shots_per_series + calibration_shots_req - 1):
            initialise_trace(False)
        else:
            cv2.waitKey(series_reset_pause * 1000)
            composite_shots = []
            shots_fired = calibration_shots_req
            initialise_trace(True)
        if  shots_fired == calibration_shots_req:
            initialise_trace(True)

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
        initialise_trace(True)
        composite_shots = []
    
    elif key_press == ord('v'):
        # Start recording
        if not record_video:
            video_start_time = time()
            video_frames = []
            record_video = True
            print('Recording started')
        else:
            # Save the video
            # Define the codec and create VideoWriter object
            video_length = time() - video_start_time
            video_fps = int(len(video_frames) / video_length)
            four_cc = cv2.VideoWriter_fourcc(*'XVID')
            video_out = cv2.VideoWriter(video_output_file, four_cc, video_fps, video_size)

            for frame in video_frames:
                video_out.write(frame)

            record_video = False
            print('Recording saved as:', video_output_file)
    
    elif key_press == ord('d'):
        # Increase debug level
        debug_level += 1
        if debug_level > debug_max:
            cv2.destroyWindow("Splatt - Blurred Vision")
            debug_level = 0
        print('Debug level:', int(debug_level))
    
    elif key_press == ord('r'):
        # Reset for recalibration
        calibration_shots = []
        composite_shots = []
        calib_XY = (0, 0)
        calibrated = False
        shots_fired = 0
        initialise_trace(True)

    elif key_press == ord('s'):
        # Save the composite image (and clear it?)
        cv2.imwrite(composite_output_file, composite_image)
    
    elif key_press == ord('f'):
        # Change the flip mode
        captured_image_flip_needed = not captured_image_flip_needed
        print('Flip required:', captured_image_flip_needed)

    elif key_press == ord(' '):
        # Undo last shot
        if (shots_fired > calibration_shots_req):
            shots_fired = max(shots_fired - 1, calibration_shots_req + 1)
            composite_shots.pop() if len(composite_shots) >= 1 else None
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)         
        else:
            shots_fired = max(shots_fired - 1, 0)
            calibration_shots.pop() if len(calibration_shots) > 1 else None