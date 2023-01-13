# Sandbox for testing sounddevice capture
# python -m sounddevice

import sounddevice as sd # requires pip install sounddevice
import numpy as np
# from numpy.fft import fft, fftfreq
from scipy.fftpack import fft, fftfreq
from matplotlib import pyplot as plt
import math
import soundfile as sf

def print_sound(indata, outdata, frames, time, status):
    volume_norm = np.linalg.norm(indata)*10
    print (int(volume_norm))

CHUNK = 44100
print(sd.query_devices()) # Choose device numbers from here. TODO: Get/save config

stream = sd.Stream(
  device=(1, 4),
  samplerate=44100,
  channels=1,
  blocksize=CHUNK)

stream.start()

assert stream.active

a = True

while a:

    indata, overflowed = stream.read(CHUNK)
    volume_norm = np.linalg.norm(indata)*10
    if int(volume_norm) > 60:
        print(int(volume_norm))
    a = False

low = 100
high = 2000
columns = 80
gain = 10
delta_f = (high - low) / columns
fftsize = math.ceil(44100 / 1000)
low_bin = math.floor(low / delta_f)

# magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
magnitude = np.abs(np.fft.fft(indata, n = fftsize))
magnitude *= gain / fftsize

sf.write('out.wav', indata, 44100)

print(magnitude)

plt.plot(abs(magnitude))
plt.show()

exit()

# plt.plot(indata)
# plt.show()
b = [(ele/2**8.)*2-1 for ele in indata]
c = fft(b)
d = len(c) / 2
# plt.plot(abs(c[:(d-1)]),'r') 
# plt.ylim(0, 1)
plt.plot(abs(c[:int(d-1)]))
plt.show()

exit()

# fft_x = fftfreq(CHUNK, 1 / 44100)
N = len(fft_y)
n = np.arange(N)
T = N / 44100
freq = n / T

plt.plot(freq, np.abs(fft_y))
plt.show()

stream.close()

nt = [(ele*2) - 1 for ele in jdata]

print(np.min(nt), np.max(nt))

X = fft(nt)
N = len(X)
n = np.arange(N)
T = N / 44100
freq = n / T

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)

plt.tight_layout()
plt.show()