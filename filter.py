import ffmpeg as ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
import contextlib
import wave
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

def get_start_end_frames(nFrames, sampleRate, tStart=None, tEnd=None):

    if tStart and tStart*sampleRate<nFrames:
        start = tStart*sampleRate
    else:
        start = 0

    if tEnd and tEnd*sampleRate<nFrames and tEnd*sampleRate>start:
        end = tEnd*sampleRate
    else:
        end = nFrames

    return (start,end,end-start)

def extract_audio(fname, tStart=None, tEnd=None):
    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        startFrame, endFrame, segFrames = get_start_end_frames(nFrames, sampleRate, tStart, tEnd)

        # Extract Raw Audio from multi-channel Wav File
        spf.setpos(startFrame)
        sig = spf.readframes(segFrames)
        spf.close()

        channels = interpret_wav(sig, segFrames, nChannels, ampWidth, True)

        return (channels, nChannels, sampleRate, ampWidth, nFrames)

def convert_to_mono(channels, nChannels, outputType):
    if nChannels == 2:
        samples = np.mean(np.array([channels[0], channels[1]]), axis=0)  # Convert to mono
    else:
        samples = channels[0]

    return samples.astype(outputType)

def fir_high_pass(samples, fs, fH, N, outputType):
    # Referece: https://fiiir.com

    fH = fH / fs

    # Compute sinc filter.
    h = np.sinc(2 * fH * (np.arange(N) - (N - 1) / 2.))
    # Apply window.
    h *= np.hamming(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[int((N - 1) / 2)] += 1
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s

def plot_audio_samples(title, samples, sampleRate, tStart=None, tEnd=None):
    if not tStart:
        tStart = 0

    if not tEnd or tStart>tEnd:
        tEnd = len(samples)/sampleRate

    f, axarr = plt.subplots(2, sharex=True, figsize=(20,10))
    axarr[0].set_title(title)
    axarr[0].plot(np.linspace(tStart, tEnd, len(samples)), samples)
    #get_specgram(axarr[1], samples, sampleRate, tStart, tEnd)

    axarr[0].set_ylabel('Amplitude')
    plt.xlabel('Time [sec]')

    plt.show()
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
    plt.xlim([100, samplerate / 2])
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.plot(freqs[:int(freqs.size / 2)], fftabs[:int(freqs.size /2)])
    plt.show()
def fir_low_pass(samples, fs, fL, N, outputType):


    fL = fL / fs

    # Compute sinc filter.
    h = np.sinc(2 * fL * (np.arange(N) - (N - 1) / 2.))
    # Apply window.
    h *= np.hamming(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s

def fir_band_pass(samples, fs, fL, fH, NL, NH, outputType):

    fH = fH / fs
    fL = fL / fs

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (np.arange(NH) - (NH - 1) / 2.))
    hlpf *= np.blackman(NH)
    hlpf /= np.sum(hlpf)
    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (np.arange(NL) - (NL - 1) / 2.))
    hhpf *= np.blackman(NL)
    hhpf /= np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((NL - 1) / 2)] += 1
    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s


####################################################################
#مرحله اول
tStart=0
tEnd=20
#
channels, nChannels, sampleRate, ampWidth, nFrames = extract_audio('monow/soot.wav', tStart, tEnd)
samples = convert_to_mono(channels, nChannels, np.int16)
#####################
#shift
samples_filtered = fir_band_pass(samples, sampleRate,0,44100,70,70, np.int16)
samples_filtered = samples_filtered  # Sound amplification
sampleRate +=15000
wavfile.write('soot15000.wav', sampleRate , samples_filtered)
stshow('soot15000.wav')
######################################################################
######################################################################
#مرحله دوم برای اعمال فیلتر حذف
# channels, nChannels, sampleRate, ampWidth, nFrames = extract_audio('salam1+soot.wav', tStart, tEnd)
# samples = convert_to_mono(channels, nChannels, np.int16)
# lp_samples_filtered = fir_low_pass(samples, sampleRate, 70,461, np.int16)               # First pass
# lp_samples_filtered = fir_low_pass(lp_samples_filtered, sampleRate, 70, 461, np.int16)   # Second pass
#
# hp_samples_filtered = fir_high_pass(samples, sampleRate, 1500, 461, np.int16)             # First pass
# hp_samples_filtered = fir_high_pass(hp_samples_filtered, sampleRate,1500, 461, np.int16) # Second pass
#
# samples_filtered = np.mean(np.array([lp_samples_filtered, hp_samples_filtered]), axis=0).astype(np.int16)
# plot_audio_samples("Sultans of Swing - After Filtering 1", samples_filtered, sampleRate, tStart, tEnd)
#
# wavfile.write('sultans_novoice1.wav', sampleRate, samples_filtered)


##############################################################################
# افزایشی
# channels, nChannels, sampleRate, ampWidth, nFrames = extract_audio('sultans_novoice1.wav', tStart, tEnd)
# samples = convert_to_mono(channels, nChannels, np.int16)
# lp_samples_filtered = fir_low_pass(samples, sampleRate, 44100,461, np.int16)               # First pass
# lp_samples_filtered = fir_low_pass(lp_samples_filtered, sampleRate, 441000, 461, np.int16) *4   # Second pass
#
# hp_samples_filtered = fir_high_pass(samples, sampleRate, 66000, 461, np.int16)             # First pass
# hp_samples_filtered = fir_high_pass(hp_samples_filtered, sampleRate,66000, 461, np.int16) # Second pass
#
# samples_filtered = np.mean(np.array([lp_samples_filtered, hp_samples_filtered]), axis=0).astype(np.int16)
# plot_audio_samples("Sultans of Swing - After Filtering 1", samples_filtered, sampleRate, tStart, tEnd)
#
# wavfile.write('nosooot2.wav', sampleRate, samples_filtered *2)
# ######################################################################
# lp_samples_filtered = fir_low_pass(samples, sampleRate, 300, 461, np.int16)               # First pass
# lp_samples_filtered = fir_low_pass(lp_samples_filtered, sampleRate, 250, 461, np.int16)   # Second pass
#
# hp_samples_filtered = fir_high_pass(samples, sampleRate, 6600, 461, np.int16)             # First pass
# hp_samples_filtered = fir_high_pass(hp_samples_filtered, sampleRate, 6600, 461, np.int16) # Second pass
#
# samples_filtered = np.mean(np.array([lp_samples_filtered, hp_samples_filtered]), axis=0).astype(np.int16)
#
# plot_audio_samples("Sultans of Swing - After Filtering 1", samples_filtered, sampleRate, tStart, tEnd)
#
# wavfile.write('sultans_novoice1.wav', sampleRate, samples_filtered)