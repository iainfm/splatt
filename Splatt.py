# Splatt! Target shooting training system
# Author: Iain McLaren
# Discussion at https://forum.stirton.com/index.php?/topic/10536-diy-scatt/ (registration required)

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
import datetime
import colorsys

# Functions TODO: move to separate file

def initialise_trace(clear_composite: bool):
    # Clear the trace and reload target_image
    global stored_trace, recorded_shot_loc, line_colour, shot_fired, target_image, composite_image, auto_reset_time_expired, this_shot_time
    stored_trace = []
    recorded_shot_loc = ()
    line_colour = []
    shot_fired = False
    target_image = blank_target_image.copy()
    if clear_composite:
        composite_image = blank_target_image.copy()
    auto_reset_time_expired = False
    this_shot_time = time()

# Audio callback function
def process_audio(indata, frames, time, status):
    global shot_detected
    if any(indata):
        audio_data = indata[:, 0]
        max_volume = np.max(np.abs(audio_data))
        if (max_volume > audio_trigger_threshold):
            shot_detected = True
        else:
            shot_detected = False

# video capture object
video_capture = cv2.VideoCapture(video_capture_device, cv2.CAP_DSHOW)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, video_frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, video_frame_height)

# Check the video stream started ok
try:
    assert video_capture.isOpened()
except:
    print('Video capture device not found. Please try changing video_capture_device in config.py')
    exit()

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

# Shot tracking
shot_fired = False
shots_fired = 0
calibrated = True if skip_calibration else False
shot_detected = False
paused = False

def setup_targets(video_width, video_height):
    target_file_image = cv2.imread(target_folder + '/' + target_filename)

    # Target image should be square for scaling etc to work properly
    print('Warning: Target image is not square') if ( target_file_image.shape[0] != target_file_image.shape[1]) else None

    target_file_image = cv2.resize(target_file_image, (video_height, video_height))   # Resize the target image to fit the video frame
    blank_target_image = np.zeros([video_height, video_width, 3], dtype = np.uint8)   # Create a new blank target image
    blank_target_image[:, :] = target_file_image[1:2, 1:2]                            # And colour it the same as the bottom-left (is it?) pixel of the target file
    target_offset = int(blank_target_image.shape[1]/2 - target_file_image.shape[1]/2) # Calculate the horizontal offset for centering the target image within the frame
    blank_target_image[0:target_file_image.shape[1], target_offset:target_file_image.shape[0] + target_offset] = target_file_image # Copy the target file into the blank target image
    return target_file_image, blank_target_image

# Create the target bitmaps
target_file_image, blank_target_image = setup_targets(video_width, video_height)

initialise_trace(True)

def draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image_old):
    # Draw the shots taken so far on the composite image
    global video_width, video_height, average_shot_loc, composite_image
    shots_plotted = 0

    for recorded_shot in composite_shots:

        calibrated_shot = convert_coordinates(recorded_shot, video_size, calibrated)

        shot_text = 'Shot ' + str(shots_plotted + 1) + ':' + str(calculate_shot_score(calibrated_shot, video_width, video_height))
        cv2.putText(composite_image, shot_text, (5, 50 + (25 * shots_plotted)), font, 1, (0, 0, 0), 1, 1)
        
        # Plot the circle on a copy of the image, then merge it with the composite_shot image to give a transparent effect
        # copy_image = composite_image.copy()
        # shot_colour = (255, 127, 0)
        cv2.circle(composite_image, calibrated_shot, scaled_shot_radius, composite_colour[shots_plotted], -1)
        # cv2.circle(copy_image, calibrated_shot, scaled_shot_radius, shot_colour, -1)
        # cv2.imshow('Splatt - Copy', copy_image)
        # alpha = 0.5
        # composite_image = cv2.addWeighted(copy_image, alpha, composite_image, 1 - alpha, 0)
        # cv2.imshow('Splatt - Composite', composite_image)
        # Draw a black outline around the shot
        # cv2.circle(composite_image, calibrated_shot, scaled_shot_radius, (0, 0, 0), 1)
        # cv2.imshow('Splatt - Copy', copy_image)
        shots_plotted += 1
                        
        # Number the shot on the composite image
        text_size = cv2.getTextSize(str(shots_plotted), font, font_scale, font_thickness)[0]
        text_X = int((calibrated_shot[0] - (text_size[0]) / 2))
        text_Y = int((calibrated_shot[1] + (text_size[1]) / 2))

        cv2.putText(composite_image, str(shots_plotted), 
                    (text_X, text_Y),
                    font, font_scale, (0,0,0), font_thickness, line_type)
        
        # Calculate the average position of the shots
        average_shot_loc = np.mean(composite_shots, axis=0)

        # Display the average position of the shots
        if len(composite_shots) > 1:
            average_centre = convert_coordinates(average_shot_loc, video_size, calibrated)
            cv2.circle(composite_image, (average_centre), int(scaled_shot_radius / 2), (255, 0, 0), 2)
            # Draw crosshairs
            cv2.line(composite_image, (average_centre[0] - scaled_shot_radius, average_centre[1]), (average_centre[0] + scaled_shot_radius, average_centre[1]), (255, 0, 0), 2)
            cv2.line(composite_image, (average_centre[0], average_centre[1] - scaled_shot_radius), (average_centre[0], average_centre[1] + scaled_shot_radius), (255, 0, 0), 2)
            real_x_offset = convert_to_real(average_centre[0] - (video_width / 2), target_diameter, video_height)
            real_y_offset = convert_to_real(average_centre[1] - (video_height / 2), target_diameter, video_height)
            real_r_offset = ((real_x_offset ** 2 + real_y_offset ** 2) ** 0.5)
            # print('Average shot offset: X:', real_x_offset, 'Y:', real_y_offset, 'R:', real_r_offset)
            offset_x = "+" if real_x_offset > 0 else "-"
            offset_y = "+" if real_y_offset < 0 else "-"
            offset_text = "Offset (mm): " + offset_x + str(abs(np.around(real_x_offset, 2))) + ", " + offset_y + str(abs(np.around(real_y_offset, 2)))
            # print(offset_text)
            cv2.putText(composite_image, offset_text, (5, video_height - 5), font, 1, (0, 0, 0), 1, 1)

def draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image):
    
    if len(composite_shots) > 1:
        # Convert the shots to the scaled and calibrated coordinates
        scaled_composite_shots = []
        for each_shot in composite_shots:
            each_shot = convert_coordinates(each_shot, video_size, calibrated)
            scaled_composite_shots.append(each_shot)

        # Find the bounding circle of all shots so far
        bc, bc_radius = cv2.minEnclosingCircle(np.asarray(scaled_composite_shots))
        bc = np.rint(bc).astype(int)

        cv2.circle(composite_image, bc, int(bc_radius + scaled_shot_radius), (0, 255, 255), 2)
        actual_spread = convert_to_real(2 * bc_radius, target_diameter, video_height) # (mm)
        cv2.putText(composite_image, 'Spread: ' + str(np.around(actual_spread, 2)) + 'mm', (5, 25), font, 1, (0, 0, 0), 1, 1)

def calculate_shot_score(shot_loc, video_width, video_height):
    # Determine the score of the shot by scoring ring diameter / shot position / bullet size
    real_x = convert_to_real(shot_loc[0] - (video_width / 2), target_diameter, video_height)
    real_y = convert_to_real(shot_loc[1] - (video_height / 2), target_diameter, video_height)
    real_r = ((real_x * real_x + real_y * real_y) ** 0.5)
    if target_scoring_scheme == 'outward':
        real_r += (shot_calibre / 2)
    else:
        real_r -= (shot_calibre / 2)

    score = 0    
    for i, scoring_ring in enumerate(target_scoring_rings):
        if real_r <= scoring_ring / 2:
            score = 10 - i
            break

    return score

def convert_coordinates(shot_loc, video_size, post_calibration: bool):
    # Convert the raw shot location to coordinates based on the scale factor and calibration offset
    coords = shot_loc
    
    # Subtract the calibration offset
    coords = np.subtract(coords, calib_XY)

    # Only scale the shot once calibration is complete
    if post_calibration:
        coords = np.subtract(coords, np.divide(video_size, 2))
        coords = np.multiply(coords, scale_factor)
        coords = np.add(coords, np.divide(video_size, 2))

    # Convert to integers
    coords = np.rint(coords).astype(int)
    return coords

def mouse_callback(event, x, y, flags, param):
    global calib_XY, composite_image, video_width, video_height
    if event == cv2.EVENT_LBUTTONDOWN:
        # calculate hue using distance formula from center of image
        # hue = np.sqrt((x - video_width / 2) ** 2 + (y - video_height / 2) ** 2) / (video_width / 2)
        # (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # R, G, B = int(255 * r), int(255 * g), int(255 * b)
        # cv2.circle(composite_image, (x, y), 20, (R, G, B), -1)

        # for x in range(1, video_width):
        #         # hue = np.sqrt((x - video_width / 2) ** 2 + (y - video_height / 2) ** 2) / (video_width / 2)
        #         hue = 0.15 + abs(x - (video_width / 2)) / video_width
        #         (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        #         R, G, B = int(255 * r), int(255 * g), int(255 * b)
        #         cv2.line(composite_image, (x, 0), (x, video_height), (R, G, B), 1)
        
        dist = np.sqrt((x - video_width / 2) ** 2 + (y - video_height / 2) ** 2)
        hue = 0.65 - ( abs(dist - (video_width / 2)) / video_width )
        (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(composite_image, (x, y), 20, (R, G, B), -1)

################################################## Main Loop ##################################################

with sd.InputStream(samplerate = audio_chunk_size, channels = 1, device = None, callback = process_audio, blocksize = 4410):
    while True: # Get the video frame
        video_ret, video_frame = video_capture.read()
        try:
            captured_image = video_frame.copy()
        except:
            print('Video frame copy failed.')

        # If set, flip the image
        if captured_image_flip_needed:
            captured_image = cv2.flip(captured_image, captured_image_flip_mode)

        # Grey-ify the image and apply a Gaussian blur to the image
        grey_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
        grey_image = cv2.GaussianBlur(grey_image, (blur_radius, blur_radius), 0)

        # Find the point of max brightness
        (min_brightness, max_brightness, min_loc, max_loc) = cv2.minMaxLoc(grey_image)

        # If minimum brightness not met skip the rest of the loop
        if max_brightness > detection_threshold:
            
            max_loc = (np.clip(max_loc[0], 0, video_width), np.clip(max_loc[1], 0, video_height))
            
            # Add the discovered point to our list with the initial line colour
            stored_trace.append(max_loc)
            line_colour.append(init_line_colour)
        else:
            # Add an on-screen indicator to show the range is not hot
            cv2.circle(target_image, (video_width - 30, video_height - 30), 20, (127, 127, 127), -1)

        # Append the new frame
        video_frames.append(target_image.copy()) if record_video else None

        # Plot the line traces so far
        for n in range(1, len(stored_trace)):
            this_line_colour = list(line_colour[n])

            if not shot_fired:

                # Check audio level if we're not paused
                if shot_detected and not paused:
                    recorded_shot_loc = max_loc
                    shot_fired = True
                    shots_fired += 1
                    shot_time = time() + auto_reset_time
                    shot_detcted = False # Reset

                # Change the colour of the traces based on colour_change_rate[]
                for c in range (0,3):
                    this_line_colour[c] = this_line_colour[c] + colour_change_rate[c]
                    if this_line_colour[c] > 255:
                        this_line_colour[c] = 255
                    elif this_line_colour[c] < 0:
                        this_line_colour[c] = 0
                line_colour[n] = tuple(this_line_colour)
            else:
                auto_reset_time_expired = True if (time() > shot_time and not paused) else False

            # Draw a line from the previous point to this one 
            line_start = convert_coordinates(stored_trace[n-1], video_size, calibrated)
            line_end = convert_coordinates(stored_trace[n], video_size, calibrated)
            cv2.line(target_image, line_start, line_end, line_colour[n], line_thickness)

        # Add an on-screen indicator to show the range is hot
        on_bull = True if calculate_shot_score(convert_coordinates(max_loc, video_size, calibrated), video_width, video_height) == 10 else False
        indicator_colour = (0, 255, 0) if on_bull else (0, 0, 255)
        
        # Clear the circle area (might be quicker to draw a square over it?)
        bg_colour = (int(target_image[1,1][0]), int(target_image[1,1][1]), int(target_image[1,1][2]))
        cv2.circle(target_image, (video_width - 30, video_height - 30), 20, bg_colour, -1)

        # Draw the indicator circle unless paused
        cv2.circle(target_image, (video_width - 30, video_height - 30), 20, indicator_colour, -1) if not paused else None

        # Give it a black outline
        cv2.circle(target_image, (video_width - 30, video_height - 30), 20, (0, 0, 0), 1)

        # Draw the shot circle if the shot has been taken TODO: Check whether it looks better to plot this under the trace
        if recorded_shot_loc:

            # Draw the shot on the target image
            calibrated_shot_loc = convert_coordinates(recorded_shot_loc, video_size, calibrated)
            cv2.circle(target_image, calibrated_shot_loc, scaled_shot_radius, shot_colour, -1)

            # Store the shot in the composite_shots list
            composite_shots.append(recorded_shot_loc)

            # Give it a random colour (old method)
            # random_colour = (random.randint(1, 64) * 4 - 1, random.randint(32, 64) * 4 - 1, random.randint(1, 64) * 4 - 1)
            # Colour it based on where it is on the target
            # calculate hue using distance formula from center of image
            # hue = np.sqrt((calibrated_shot_loc[0] - video_width / 2) ** 2 + (calibrated_shot_loc[1] - video_height / 2) ** 2) / (video_height / 2)
            dist = np.sqrt((calibrated_shot_loc[0] - video_width / 2) ** 2 + (calibrated_shot_loc[1] - video_height / 2) ** 2)
            hue = 0.65 - (abs(dist - (video_width / 2)) / video_width)
            (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            stored_colour = (int(255 * r), int(255 * g), int(255 * b))
            composite_colour.append(stored_colour)

            # Clear the composite image and redraw the shots
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)
                
            # TODO: anything else required with the recorded shot location (eg csv output)
            # Reset for the next shot             
            recorded_shot_loc = ()

        if not calibrated:
            cv2.putText(target_image, 'CALIBRATING', (5, 25), font, 2, (0, 0, calib_text_red), 1, 1)
            
            # Fancy calibration font colour
            calib_text_red += d_calib_text_red
            if calib_text_red > 255:
                calib_text_red = 255
                d_calib_text_red = -8
            elif calib_text_red < calib_text_red_min:
                calib_text_red = calib_text_red_min
                d_calib_text_red = 8

        else:
            if calibrated and display_shot_time:
                text = "%04.2f" % (time() - this_shot_time, )
                text_size = cv2.getTextSize(text, font, 2, font_thickness)[0]
                cv2.rectangle(target_image, (3, 23), (7 + text_size[0], 27 + text_size[1]), (0, 0, 0), -1)
                cv2.putText(target_image, text, (5, 26 + text_size[1]), font, 2, (255, 255, 255), 1, 1, False)

        # Show the images
        cv2.imshow('Splatt - Live trace', target_image)
        cv2.imshow('Splatt - Composite', composite_image)
        cv2.imshow('Splatt - Blurred Vision', grey_image) if debug_level > 0 else None

        # Testing mouse operations
        cv2.setMouseCallback('Splatt - Composite', mouse_callback)

        if auto_reset_time_expired:
            if (shots_fired <= shots_per_series - 1):
                initialise_trace(False)
            else:
                cv2.waitKey(series_reset_pause * 1000)
                composite_shots = []
                shots_fired = 0
                initialise_trace(True)

        # Check for user input
        key_press = cv2.waitKeyEx(1)
        
        if key_press == ord('.'):
            if debug_level > 0:
                print('calib_XY:', calib_XY)
                print('scale_factor:', scale_factor)
                print('max_loc:', max_loc)

        elif key_press == 13: # User has finished calibrating
            if (shots_fired > 0) or (skip_calibration == True): # Do we still need to test for this?
                calibrated = True

                # Find the spread of calibration shots
                (cal_X, cal_Y), cal_radius = cv2.minEnclosingCircle(np.asarray(composite_shots))

                # Add the calibration offset to any manual adjustments
                calib_XY = np.add(calibrate_offset(video_width, video_height, average_shot_loc[0], average_shot_loc[1]), calib_XY)

                # Plot a circle to show the shot spread
                cv2.circle(target_image, ((int(cal_X), int(cal_Y))), int(cal_radius), (0, 255, 255), 2)
                cv2.circle(target_image, ((int(cal_X), int(cal_Y))), 2, (0, 255, 255), 2)

                # Reset the range
                shots_fired = 0
                composite_shots = []
                initialise_trace(True)
        
        elif key_press == ord(' '): # Undo last shot
            shots_fired = max(shots_fired - 1, 0)
            composite_shots.pop() if len(composite_shots) >= 1 else None
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)    

        elif (key_press | 32) == ord('c'): # Clear the trace            
            initialise_trace(True)
            composite_shots = []

        elif (key_press | 32) == ord('d'): # Increase debug level
            debug_level += 1
            if debug_level > debug_max:
                cv2.destroyWindow("Splatt - Blurred Vision")
                debug_level = 0
            print('Debug level:', int(debug_level))

        elif (key_press | 32) == ord('f'): # Change the flip mode
            captured_image_flip_needed = not captured_image_flip_needed
            print('Flip required:', captured_image_flip_needed)

        elif (key_press | 32) == ord('g'): # Change the flip axis
            captured_image_flip_mode += 1
            if captured_image_flip_mode > 1:
                captured_image_flip_mode = -1
            print('Flip mode:', captured_image_flip_mode)

        elif (key_press | 32) == ord('p'): # Pause
            paused = not paused
            print('Paused:', paused) if debug_level > 0 else None

        elif (key_press | 32) == ord('q'): # Quit
            video_capture.release()
            exit()

        elif (key_press | 32) == ord('r'): # Reset for recalibration
            composite_shots = []
            calib_XY = (0, 0)
            calibrated = False
            shots_fired = 0
            initialise_trace(True)

        elif (key_press | 32) == ord('s'): # Save the composite image (and clear it?)
            t = datetime.datetime.now()
            cv2.imwrite(composite_output_file + '-' + t.strftime("%Y%m%d-%H%M%S") + '.png', composite_image)
        
        elif (key_press | 32) == ord('v'): # Video recording
            if not record_video: # Start recording
                video_start_time = time()
                video_frames = []
                record_video = True
                print('Recording started')
            else: # Save the video
                video_length = time() - video_start_time
                video_fps = int(len(video_frames) / video_length)
                four_cc = cv2.VideoWriter_fourcc(*'XVID')
                t=datetime.datetime.now()
                video_out = cv2.VideoWriter(video_output_file + '-' + t.strftime("%Y%m%d-%H%M%S") + '.avi', four_cc, video_fps, video_size)

                for frame in video_frames:
                    video_out.write(frame)

                record_video = False
                print('Recording saved as:', video_output_file + '-' + t.strftime("%Y%m%d-%H%M%S") + '.avi')
        
        elif (key_press == ord('2') or key_press == 2621440): # Down arrow - adjust calibration down
            calib_XY = (calib_XY[0], calib_XY[1] - 2)
            print('calib_XY:', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            target_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)

        elif (key_press == ord('4') or key_press == 2424832): # Left arrow - adjust calibration left
            calib_XY = (calib_XY[0] + 2, calib_XY[1])
            print('calib_XY:', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            target_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)

        elif (key_press == ord('6') or key_press == 2555904): # Right arrow - adjust calibration right
            calib_XY = (calib_XY[0] - 2, calib_XY[1])
            print('calib_XY:', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            target_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)

        elif (key_press == ord('8') or key_press == 2490368): # Up arrow - adjust calibration up
            calib_XY = (calib_XY[0], calib_XY[1] + 2)
            print('calib_XY:', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            target_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)
        
        elif (key_press == ord('7') or key_press == 2228224): # Page down - decrease the scale factor
            if real_range_length > 1:
                real_range_length -= 1
            scale_factor -= 0.05
            if scale_factor < 0.05:
                scale_factor = 0.05
            print('Scale factor:', scale_factor) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            target_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)

        elif (key_press == ord('9') or key_press == 2162688): # Page up - Increase the scale factor
            scale_factor += 0.05
            print('Scale factor:', scale_factor) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            target_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)