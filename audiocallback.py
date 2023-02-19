#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

PySoundFile (https://github.com/bastibe/PySoundFile/) has to be installed!

"""

import tempfile
import queue
import sys
import numpy as np


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text



try:
    import sounddevice as sd
    import soundfile as sf


    q = queue.Queue()
    args_filename = "out.wav"
    args_samplerate = 44100
    args_device = None
    args_channels = 1
    args_subtype = None

    def callback(indata, frames, time, status):
        if any(indata):
            audio_data = indata[:, 0]
            max_volume = np.max(np.abs(audio_data))
            audio_data_normalised = audio_data / np.max(audio_data)
            fft_data = np.abs(np.fft.rfft(audio_data_normalised))
            if (max_volume > 0.01):
                try:
                    print(max_volume, '\t', np.argmax(fft_data), len(fft_data))
                except KeyboardInterrupt:
                    exit()

    with sd.InputStream(samplerate=args_samplerate, device=args_device,
                        channels=args_channels, callback=callback):
        print('#' * 80)
        print('press Ctrl+C to stop the recording')
        print('#' * 80)
        while True:
            a = 1

except KeyboardInterrupt:
    exit()