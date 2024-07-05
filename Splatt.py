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
        # audio_data_normalised = audio_data / np.max(audio_data)
        # fft_data = np.abs(np.fft.rfft(audio_data_normalised))
        # volume_norm = np.linalg.norm(audio_data) * 10
        # print(max_volume, '\t', np.argmax(fft_data), len(fft_data)) if debug_level > 0 else None
        if (max_volume > audio_trigger_threshold):
            # TODO: Tune this and add clause for FFT result
            shot_detected = True
        else:
            shot_detected = False

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

# Shot tracking
shot_fired = False
shots_fired = 0
calibrated = False
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

# Start listening and check it started
# audio_stream.start()
# assert audio_stream.active

initialise_trace(True)

def draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image):
    global video_width, video_height
    shots_plotted = 0
    for recorded_shot in composite_shots:
        calibrated_shot = np.add(recorded_shot, calib_XY) # [0] + calib_XY[0], recorded_shot[1] + calib_XY[1])
        shot_text = 'Shot ' + str(shots_plotted + 1) + ':' + str(calculate_shot_score(calibrated_shot, video_width, video_height))
        cv2.putText(composite_image, shot_text, (5, 50 + (25 * shots_plotted)), font, 1, (0, 0, 0), 1, 1)
        shots_plotted += 1
        composite_colour = (random.randint(1, 64) * 4 - 1, random.randint(32, 64) * 4 - 1, random.randint(1, 64) * 4 - 1)
        cv2.circle(composite_image, calibrated_shot, scaled_shot_radius, composite_colour, line_thickness)
                
        # Number the shot on the composite image
        text_size = cv2.getTextSize(str(shots_plotted), font, font_scale, font_thickness)[0]
        text_X = int((calibrated_shot[0] - (text_size[0]) / 2))
        text_Y = int((calibrated_shot[1] + (text_size[1]) / 2))

        cv2.putText(composite_image, str(shots_plotted), 
                    (text_X, text_Y),
                    font, font_scale, composite_colour, font_thickness, line_type)

def draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image):
    # Find the bounding circle of all shots so far
    if len(composite_shots) > 1:
        (bc_X, bc_Y), bc_radius = cv2.minEnclosingCircle(np.asarray(composite_shots))
        bc_centre = (int(bc_X + calib_XY[0]), int(bc_Y + calib_XY[1]))
        cv2.circle(composite_image, bc_centre, int(bc_radius + scaled_shot_radius), (0, 255, 255), 2)
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

################################################## Main Loop ##################################################

with sd.InputStream(samplerate = audio_chunk_size, channels = 1, device = None, callback = process_audio, blocksize = 4410):
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

            # Offset the location based on the calibration
            max_loc_x = int(( max_loc[0] )) # + calib_XY[0] ) )
            max_loc_y = int(( max_loc[1] )) # + calib_XY[1] ) )

            if calibrated:
                # Scale based on the virtual / real range distances, limited to video frame dimensions
                max_loc_x = np.clip(int( scale_factor * (max_loc_x - video_width / 2) + video_width / 2 ), 0, video_width)
                max_loc_y = np.clip(int( scale_factor * (max_loc_y - video_height / 2) + video_height / 2 ), 0, video_height)
            
            max_loc = (max_loc_x, max_loc_y)
            
            # Add the discovered point to our list with the initial line colour
            stored_trace.append((max_loc_x, max_loc_y))
            line_colour.append(init_line_colour)
        else:
            # Add an on-screen indicator to show the range is not hot
            cv2.circle(target_image, (video_width - 30, video_height - 30), 20, (127, 127, 127), -1)

        # Append the new frame
        video_frames.append(target_image.copy()) if record_video else None

        target_image = blank_target_image.copy()

        # Plot the line traces so far
        for n in range(1, len(stored_trace)):
            this_line_colour = list(line_colour[n])

            if not shot_fired:

                # Check audio level and peak frequency if we're not paused
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
            calibrated_line_start = np.add(stored_trace[n-1], calib_XY) #, stored_trace[n-1][1] + calib_XY[1])
            calibrated_line_end = np.add(stored_trace[n], calib_XY) #, stored_trace[n][1] + calib_XY[1])
            
            cv2.line(target_image, calibrated_line_start, calibrated_line_end, line_colour[n], line_thickness)

        # Add an on-screen indicator to show the range is hot
        # thickness = -1 if not paused else 1
        indicator_thickness = -1 if not paused else 1
        on_bull = True if calculate_shot_score(max_loc, video_width, video_height) == 10 else False
        indicator_colour = (0, 255, 0) if on_bull else (0, 0, 255)

        cv2.circle(target_image, (video_width - 30, video_height - 30), 20, indicator_colour, indicator_thickness)

        # Draw the shot circle if the shot has been taken TODO: Check whether it looks better to plot this under the trace

        if recorded_shot_loc:
            calibrated_shot_loc = np.add(recorded_shot_loc, calib_XY) #, recorded_shot_loc[1] + calib_XY[1]
            cv2.circle(target_image, calibrated_shot_loc, scaled_shot_radius, shot_colour, -1)

            composite_shots.append(recorded_shot_loc)
            composite_image = blank_target_image.copy()

            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)
                
            # Remember to do anything else required with the recorded shot location here (eg csv output)

            # print(calculate_shot_score(recorded_shot_loc, video_width, video_height))   
            
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

        cv2.imshow('Splatt - Live trace', target_image)
        cv2.imshow('Splatt - Composite', composite_image)
        cv2.imshow('Splatt - Blurred Vision', grey_image) if debug_level > 0 else None

        if auto_reset_time_expired:
            if (shots_fired <= shots_per_series - 1):
                initialise_trace(False)
            else:
                cv2.waitKey(series_reset_pause * 1000)
                composite_shots = []
                shots_fired = 0
                initialise_trace(True)

        # Check for user input
        key_press = cv2.waitKey(1) & 0xFF
        
        if key_press == ord('.'):
            print('.') if debug_level > 0 else None

        elif key_press == 13:
            # User has finished calibrating
            if shots_fired > 0:
                calibrated = True
                # Find the spread of calibration shots
                (cal_X, cal_Y), cal_radius = cv2.minEnclosingCircle(np.asarray(composite_shots))
                calib_XY = calibrate_offset(video_width, video_height, cal_X, cal_Y)

                # Plot a circle to show the calibration data
                print('Calibration offset (px):', calib_XY) if debug_level >= 0 else None
                cv2.circle(target_image, ((int(cal_X), int(cal_Y))), int(cal_radius), (0, 255, 255), 2)
                cv2.circle(target_image, ((int(cal_X), int(cal_Y))), 2, (0, 255, 255), 2)

                # Reset the range
                shots_fired = 0
                composite_shots = []
                initialise_trace(True)
        
        elif key_press == ord(' '):
            # Undo last shot
            shots_fired = max(shots_fired - 1, 0)
            composite_shots.pop() if len(composite_shots) >= 1 else None
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
            draw_bounding_circle(video_height, scaled_shot_radius, font, composite_image)    

        elif key_press == ord('c'):
            # Clear the trace
            initialise_trace(True)
            composite_shots = []

        elif key_press == ord('d'):
            # Increase debug level
            debug_level += 1
            if debug_level > debug_max:
                cv2.destroyWindow("Splatt - Blurred Vision")
                debug_level = 0
            print('Debug level:', int(debug_level))

        elif key_press == ord('f'):
            # Change the flip mode
            captured_image_flip_needed = not captured_image_flip_needed
            print('Flip required:', captured_image_flip_needed)

        elif key_press == ord('p'):
            # Pause
            paused = not paused
            print('Paused:', paused) if debug_level > 0 else None

        elif key_press == ord('q'):
            # Quit
            video_capture.release()
            exit()

        elif key_press == ord('r'):
            # Reset for recalibration
            composite_shots = []
            calib_XY = (0, 0)
            calibrated = False
            shots_fired = 0
            initialise_trace(True)

        elif key_press == ord('s'):
            # Save the composite image (and clear it?)
            cv2.imwrite(composite_output_file, composite_image)
        
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
        
        elif key_press == ord('2'):
            # print('down')
            calib_XY = (calib_XY[0], calib_XY[1] + 2)
            print('Calibration offset (px):', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)

        elif key_press == ord('4'):
            # print('left')
            calib_XY = (calib_XY[0] - 2, calib_XY[1])
            print('Calibration offset (px):', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)

        elif key_press == ord('6'):
            # print('right')
            calib_XY = (calib_XY[0] + 2, calib_XY[1])
            print('Calibration offset (px):', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)

        elif key_press == ord('8'):
            # print('up')
            calib_XY = (calib_XY[0], calib_XY[1] - 2)
            print('Calibration offset (px):', calib_XY) if debug_level > 0 else None
            composite_image = blank_target_image.copy()
            draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)
        
        # TODO: Live adjustnment of the range length. More work required; shots are stored already scaled. Need to separate the scaling from the drawing
        # elif key_press == ord('7'):
        #     # Adjust the real range length
        #     real_range_length -= 1
        #     scale_factor = simulated_range_length / real_range_length
        #     print('Real range length:', real_range_length, 'm') if debug_level > 0 else None
        #     composite_image = blank_target_image.copy()
        #     draw_composite_shots(scaled_shot_radius, font, font_scale, composite_image)

        # elif key_press == ord('9'):
        #     # Adjust the real range length
        #     real_range_length += 1
        #     scale_factor = simulated_range_length / real_range_length
        #     print('Real range length:', real_range_length, 'm') if debug_level > 0 else None
        #     composite_image = blank_target_image.copy()