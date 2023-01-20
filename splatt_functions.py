# Functions used by Splatt

def calibrate_offset(video_width: int, video_height: int, max_loc_X: int, max_loc_Y: int) -> int:
    # Calibrate offset to point source
    return (int((video_width / 2) - max_loc_X), int((video_height / 2) - max_loc_Y))

def convert_to_pixels(size_in_mm, target_diameter, video_height: int) -> int:
    # Convert real-world dimensions to pixels base on target_szie:image_height
    return int(size_in_mm * video_height / target_diameter)

def convert_to_real(size_in_pixels, target_diameter, video_height:int):
    # Convert size measured in pixels to real-world dimensions based on target_size:image_height
    return size_in_pixels * target_diameter / video_height
