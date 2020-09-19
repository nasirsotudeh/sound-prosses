# sound-prosses™
### Project Description: 
 The intention of this project is to record the sound and draw a diagram of its frequency range and then apply echo on the sound.
And the frequency amplitude transmission and then the addition of the echo signal with the transmitted signal.
In this project, Python language is used, which has several libraries in the basin for drawing diagrams and taking parameters.
 # install requirements.txt
 ```
 pip install -r requirements.txt
 ```
The first step is to practically load the data into a format understandable to the machine. For this, we simply get the values ​​6 seconds after each specific time step; For example, in an audio file, we extract values ​​in half a second. This is called sampling of and its rate is the audio data sampling rate. (Called sampling rate)
Another way to display audio data is to convert it to a different representation of the data range, the frequency range. When we consider an audio data as a frequency domain, we need a lot of data points to represent the whole data, and the sampling rate should be as high as possible. On the other hand, if we display audio information as a frequency range

## Read and write WAV files using Python (wave) 
First we need to record some audio files that our audio files are in wav format.
It should be noted that each wav file can be saved in two ways:
stereo and mono
The difference between the two is in the number of frequency channels, which is known as mono sound. In it, all the different audio signals are combined to form a single audio channel, hence the name mono (single or single).
Audio signals in stereo are divided into two or more channels, each of which is sent to a different speaker. This helps to better understand the direction and depth of sound.


```
setnchannels() 
```
``
Set the number of channels. 1 for Mono 2 for stereo channels 
``

```
obj = wave.open('sound.wav','wb') 
```
This function opens a file to read / write audio data.
This function requires two parameters
first the file name and second the mode


