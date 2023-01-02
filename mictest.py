# Print out realtime audio volume as ascii bars

import sounddevice as sd # requires pip install sounddevice
import numpy as np

def print_sound(indata, outdata, frames, time, status):
    volume_norm = np.linalg.norm(indata)*10
    print (int(volume_norm))

#with sd.Stream(callback=print_sound):
#    sd.sleep(10000)

#assert False

# sd.Stream().read(100)

CHUNK = 4096

stream = sd.Stream(
  device=("Microphone Array (Realtek High , MME", "Speaker/HP (Realtek High Defini, MME"),
  samplerate=44100,
  channels=2,
  blocksize=CHUNK)

stream.start()
while True:
    indata, overflowed = stream.read(CHUNK)
    # print(indata)
    volume_norm = np.linalg.norm(indata)*10
    print(int(volume_norm))
stream.close()

# python -m sounddevices

#while True:
    #print(int(volume_norm))