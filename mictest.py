# Sandbox for testing sounddevice capture
# python -m sounddevice

import sounddevice as sd # requires pip install sounddevice
import numpy as np

def print_sound(indata, outdata, frames, time, status):
    volume_norm = np.linalg.norm(indata)*10
    print (int(volume_norm))

CHUNK = 4096
print(sd.query_devices()) # Choose device numbers from here. TODO: Get/save config

stream = sd.Stream(
  device=(1, 4),
  samplerate=44100,
  channels=1,
  blocksize=CHUNK)

stream.start()

assert stream.active

while True:
    indata, overflowed = stream.read(CHUNK)
    volume_norm = np.linalg.norm(indata)*10
    if int(volume_norm) > 30:
        print(int(volume_norm))
stream.close()
