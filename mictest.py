# Sandbox for testing sounddevice capture
# python -m sounddevice

import sounddevice as sd # requires pip install sounddevice
import numpy as np
# from numpy.fft import fft, fftfreq
# from scipy.fftpack import fft, fftfreq
# from matplotlib import pyplot as plt
# import math
import soundfile as sf
from scipy.signal import butter, lfilter

def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 5
fs = 44100.0       # sample rate, Hz
cutoff = 4000  # desired cutoff frequency of the filter, Hz
T = 1.0
CHUNK = int(T * fs)

print(sd.query_devices()) # Choose device numbers from here. TODO: Get/save config

stream = sd.Stream(
  device=(1, 3),
  samplerate=44100,
  channels=1,
  blocksize=CHUNK)

stream.start()

assert stream.active

a = True

while a:

    indata, overflowed = stream.read(CHUNK)
    ndata = np.linalg.norm(indata)
    volume_norm = np.linalg.norm(indata)*10
    if int(volume_norm) > 60:
        print(int(volume_norm))
    a = False

# indata = np.random.default_rng().uniform(-0.5, 0.5, int(T * fs))

if any(indata):
  cdata = indata[:, 0]

  print(np.min(cdata), np.max(cdata))
  sd.play(indata, fs)
  sd.wait()

  fldata = butter_highpass_filter(indata[:, 0], cutoff, fs, order)
  fldata = fldata / np.max(fldata) # Normalise

  print(np.min(fldata), np.max(fldata))

  sd.play(fldata, fs)
  sd.wait()

  sf.write('in.wav', indata, 44100)
  sf.write('fl.wav', fldata, 44100)

exit()

# Get the filter coefficients so we can check its frequency response.
# b, a = butter_highpass(cutoff, fs)
# print(b, a)

# low = 100
# high = 2000
# columns = 80
# gain = 10
# delta_f = (high - low) / columns
# fftsize = math.ceil(44100 / 1000)
# low_bin = math.floor(low / delta_f)

# magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
# magnitude = np.abs(np.fft.fft(indata, n = fftsize))
# magnitude *= gain / fftsize

# print(magnitude)

# plt.plot(abs(magnitude))
# plt.show()



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