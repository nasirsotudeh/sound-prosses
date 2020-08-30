import wave
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import wave
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq
import sys

# mono file
def monoshow(f):
    # Load the data and calculate the time of each sample
    samplerate, data = f
    times = np.arange(len(data)) / float(samplerate)
    plt.figure(figsize=(15, 10))
    plt.fill_between(times, data)
    plt.xlim(times[1000], times[-1])
    plt.xlabel('time (s) mono')
    plt.ylabel('amplitude')
    # You can set the format by changing the extension
    #  like .pdf, .svg, .eps
    plt.savefig('plot.png', dpi=1000)
    plt.show()

def stereoshow(f):
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    time = np.arange(0, nframes) * (1.0 / framerate)
    pl.subplot(211)
    pl.plot(time, wave_data[1], c="g")
    pl.xlabel("time (seconds)")
    pl.show()


def stshow(file):

    samplerate, data = wavfile.read(file)
    samples = data.shape[0]
    plt.plot(data[:200])

    datafft = fft(data)
    # Get the absolute value of real and complex component:
    fftabs = abs(datafft)

    freqs = fftfreq(samples, 1 / samplerate)
    plt.plot(freqs, fftabs)
    datafft = fft(data)
    # Get the absolute value of real and complex component:
    fftabs = abs(datafft)
    freqs = fftfreq(samples, 1 / samplerate)
    plt.xlim([10, samplerate / 2])
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.plot(freqs[:int(freqs.size / 2)], fftabs[:int(freqs.size / 2)])
    plt.show()

def getfile(p):
    # p = 'monow/1.wav'

    obj = wave.open(p, 'r')
    print("Number of channels", obj.getnchannels())
    print("Sample width", obj.getsampwidth())
    print("Frame rate.", obj.getframerate())
    print("Number of frames", obj.getnframes())
    print("parameters:", obj.getparams())
    if (obj.getnchannels() == 1):
        obj = wavfile.read(p, 'r')
        monoshow(obj)
    else:
        stereoshow(obj)
        obj.close()

#########################################################
# Test 1 :
getfile('monow/salam1.wav')
# getfile('echsalam_1.wav')
# stshow('monow/salam1.wav')
# stshow('soot1500.wav')
# stshow('echsalam_1.wav')


# test 2:
# getfile('stereow/salam4.wav')
# getfile('out11.wav')
############################################################
def getob(p):
    stshow(p)