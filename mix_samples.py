'''
Script to mix two testing samples
'''
import librosa
import numpy as np


# provide the wav name and mix
# speech1 = '/media/nca/data/raw_data/speech_train_r/FCMM0/TRAIN_DR2_FCMM0_SI1957.WAV'
# speech2 = '/media/nca/data/raw_data/speech_train_r/FKLC0/TRAIN_DR4_FKLC0_SX355.WAV'
speech1 = '/Users/JAKE/Documents/deep-clustering/test/FA/FA01_01.wav'
speech2 = '/Users/JAKE/Documents/deep-clustering/test/MC/MC13_01.wav'

data1, _ = librosa.load(speech1, sr=8000)
data2, _ = librosa.load(speech2, sr=8000)
mix = data1[:len(data2)] + data2[:len(data1)] * 0.1
librosa.output.write_wav('mix.wav', mix, 8000)
