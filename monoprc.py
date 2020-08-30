from audioop import add
import wave
from warnings import warn
from audioop import mul


def input_wave(filename, frames=10000000):
    with wave.open(filename, 'rb') as wave_file:
        params = wave_file.getparams()
        audio = wave_file.readframes(frames)
        if params.nchannels != 1:
            raise Exception("The input audio should be mono for these examples")
    return params, audio


def output_wave(audio, params, stem, suffix):
    filename = stem.replace('.wav', '_{}.wav'.format(suffix))
    with wave.open(filename, 'wb') as wave_file:
        wave_file.setparams(params)
        wave_file.writeframes(audio)


def delay(audio_bytes, params, offset_ms):
    """version 1: delay after 'offset_ms' milliseconds"""
    # calculate the number of bytes which corresponds to the offset in milliseconds
    offset = params.sampwidth * offset_ms * int(params.framerate / 1000)
    # create some silence
    beginning = b'\0' * offset
    # remove space from the end
    end = audio_bytes[:-offset]
    return add(audio_bytes, beginning + end, params.sampwidth)


def delay(audio_bytes, params, offset_ms, factor=1, num=1):
#delays after 'offset_ms' milliseconds amplified by 'factor'
    if factor >= 1:
        warn("These settings may produce a very loud audio file. \
              Please use caution when listening")
    # calculate the number of bytes which corresponds to the offset in milliseconds
    offset = params.sampwidth * offset_ms * int(params.framerate / 1000)
    # add extra space at the end for the delays
    audio_bytes = audio_bytes + b'\0' * offset * (num)
    # create a copy of the original to apply the delays
    delayed_bytes = audio_bytes
    for i in range(num):
        # create some silence
        beginning = b'\0' * offset * (i + 1)
        # remove space from the end
        end = audio_bytes[:-offset * (i + 1)]
        # multiply by the factor
        multiplied_end = mul(end, params.sampwidth, factor ** (i + 1))
        delayed_bytes = add(delayed_bytes, beginning + multiplied_end, params.sampwidth)
    return delayed_bytes


def delay_to_file(audio_bytes, params, offset_ms, file_stem, factor=1, num=1):
    echoed_bytes = delay(audio_bytes, params, offset_ms, factor, num)
    output_wave(echoed_bytes, params, file_stem,
                '1'.format(offset_ms, factor, num))

####################################################
tr_params, tr_bytes = input_wave('monow/salam1.wav')
# Test 1:
delay_to_file(tr_bytes, tr_params, offset_ms=125, file_stem='echsalam.wav', factor=.7, num=10)
print("First 10 bytes:", tr_bytes[:10], sep='\n')



####################################################
####################################################