# Sandbox for testing sounddevice capture
# python -m sounddevice

import sounddevice as sd # requires pip install sounddevice
import numpy as np
import pickle

# from numpy.fft import fft, fftfreq
# from scipy.fftpack import fft, fftfreq
from matplotlib import pyplot as plt
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
cutoff = 10000  # desired cutoff frequency of the filter, Hz
T = 1.0
CHUNK = 4096 # int(T * fs)

click_threshold = 50

print(sd.query_devices()) # Choose device numbers from here. TODO: Get/save config

stream = sd.Stream(
  device=None,
  samplerate=44100,
  channels=1,
  blocksize=CHUNK)

stream.start()

assert stream.active

volume_norm = 0
training_mode = False

def arrays_match(arr1, arr2, tolerance):
    print('len=', len(arr1), len(arr2))
    if len(arr1) != len(arr2):
        return False
    for i in range(len(arr1)):
        print(abs(arr1[i] - arr2[i]))
        if abs(arr1[i] - arr2[i]) > tolerance:
            print('i=', i)
            return False
    return True

def waveform_similarity(waveform1, waveform2):
  mse = np.mean((waveform1 - waveform2) ** 2)
  print(mse)
  return mse

if training_mode:
  while (volume_norm < click_threshold):
      indata, overflowed = stream.read(CHUNK)
      # ndata = np.linalg.norm(indata)
      volume_norm = np.linalg.norm(indata[:, 0]) * 10
      print(int(volume_norm))
    
  # indata = np.random.default_rng().uniform(-0.5, 0.5, int(T * fs))

  if any(indata):

    sd.play(indata, fs)
    sd.wait()

    fldata = butter_highpass_filter(indata[:, 0], cutoff, fs, order)
    fldata = fldata / np.max(fldata) # Normalise

    print(np.min(fldata), np.max(fldata))

    fft_data = abs(np.fft.rfft(fldata))
    print(np.shape(fft_data))

    plt.plot(fft_data)
    plt.show()

    sd.play(fldata, fs)
    sd.wait()

    sf.write('in.wav', indata, 44100)
    sf.write('fl.wav', fldata, 44100)

    with open("sound_fingerprint.pkl", "wb") as f:
      pickle.dump(fft_data, f)
    volume_norm = 0

else:
  with open("sound_fingerprint.pkl", "rb") as f:
    fingerprint_data = pickle.load(f)
    print(np.shape(fingerprint_data))
  while True:
    while (volume_norm < click_threshold):
      indata, overflowed = stream.read(CHUNK)
      # ndata = np.linalg.norm(indata)
      volume_norm = np.linalg.norm(indata[:, 0]) * 10

    if any(indata):
      
      # fldata = butter_highpass_filter(indata[:, 0], cutoff, fs, order)
      # fldata = fldata / np.max(fldata) # Normalise
      fldata = indata[:, 0]
      # fldata = fldata / np.max(fldata)
      print(len(fldata))
      fft_data = abs(np.fft.rfft(fldata))
    
      # plt.plot(fft_data / np.max(fft_data))
      # plt.show()
      # fft_data = fft_data[100:]
      peak_frequency = np.argmax(fft_data)
      trigger_frequency = 790
      trigger_threshold = 10
      print('volume:', volume_norm, 'peak freq:', peak_frequency, 'max: ', fft_data[peak_frequency])
      
      volume_norm = 0
      if abs(peak_frequency - trigger_frequency) < trigger_threshold:
        print("Shot fired")