# Debug settings
debug_level =  0  # 0 (off), 1 (info), 2 (detailed)
debug_max = 2     # max debug level

# Hardware settings
video_capture_device = 1    # 0 is usually the first (eg built-in) camera. 1 is external (if built-in exists) TODO: make this better

# Virtual shooting range and session options
real_range_length = 20      # (units must match simulated range length)
shot_calibre = 5.6          # mm (0.22")
session_name = 'Practice 13/01/23'
auto_reset = True           # reset after shot taken
auto_reset_time = 3         # Number of seconds after the shot before resetting
# calibration_shots_req = 3 # Number of shots to average to calibrate the system (deprecated)
shots_per_series = 5        # How many shots before auto-resetting
series_reset_pause = 3      # seconds to pause before starting a new series
target_index = 3            # Target to use (see below)

# Target dimensions
# (name, diameter (mm), filename, simulated_range_length, (ring scores hi->low), gauging)
# TODO: add dimensions for scoring rings
target = (('6 yard air rifle', 31.00, '1989 6yard Inward Gauging.png', 6, (1.00, 6.00, 11.00, 16.00, 21.00, 26.00, 31.00, 36.00, 41.00, 46.00), 'inward'),
          ('10 metre air rifle', 31.20, '1989 10m Outward Gauging.png', 10, (8.80, 12.00, 15.20, 18.40, 21.60, 24.80, 28.00, 31.20, 34.40, 37.60), 'outward'),
          ('15 yard prone', 30.84, '1989 15yard Outward Gauging.png', 15, (9.99, 14.38, 18.77, 23.16, 27.55, 31.93, 36.32, 40.71, 45.10, 49.49), 'outward'),
          ('25 yard prone', 51.39, '1989 25yard Outward Gauging.png', 25, (12.92, 20.23, 27.55, 34.86, 42.18, 49.49, 56.81, 64.12, 71.44, 78.75), 'outward'),
          ('50 yard prone', 102.79, '1989 50yard Inward Gauging.png', 50, (9.03, 23.66, 38.29, 52.92, 67.55, 82.18, 96.81), 'inward'),
          ('100 yard prone', 205.55, '1989 100yard Inward Gauging.png', 100, (26.48, 57.32, 88.16, 119.00, 149.84, 180.68, 211.52, 242.36, 273.20), 'inward'))

target_name = target[target_index][0]
target_diameter = target[target_index][1]
target_filename = target[target_index][2]
simulated_range_length = target[target_index][3]
target_scoring_rings = target[target_index][4]
target_scoring_scheme = target[target_index][5]
target_folder = 'Targets'

# Scaling variables
scale_factor = simulated_range_length / real_range_length
  
# video capture object
captured_image_flip_needed = False # Whether the camera is mounted upside down
video_frames = []

# Recording options
record_video = False # (Do not set to true here)
video_output_file = 'output.avi'
composite_output_file = 'composite.png'

# Audio and video processing options
blur_radius = 11          # must be an odd number, or else GaussianBlur will fail. Lower is better for picking out point sources
detection_threshold = 50  # Trigger value to detect the reference point
click_threshold = 50      # audio level that triggers a 'shot'

# Plotting colours and options
init_line_colour = (0, 0, 255, 0) # (Blue, Green, Red)
shot_colour = (255, 0, 255)       # Magenta
line_thickness = 2
colour_change_rate = (0, 15, -15) # Rates of colour change per frame (b, g, r)
display_shot_time = True # Overlay a timer onto the trace

# Initialise tuples and variables
composite_shots = []
calibration_shots = []

# Coordinates of the 'fired' shot
recorded_shot_loc = []

# Calibration / scaling
calib_XY = (0, 0)

# Font details
font_thickness = 1
line_type = 1

# Fancy printing of 'calibration'
calib_text_red = 255
d_calib_text_red = -2
calib_text_red_min = 127

# Sound capture parameters
audio_chunk_size = 4410

# Audio fingerprint of a shot being fired (FFT analysis from mictest.py)
audio_trigger_frequency = 300 # Anschutz 1813 dryfiring empty cases =~ 298
audio_trigger_threshold = 0.1