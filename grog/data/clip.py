import glob
import librosa
import os
import numpy as np
import argparse
import math
from multiprocessing import Pool
from grog.config import Config
import soundfile as sf


def clip(data_dir, N, low, high, duration, output_dir):
    speakers_dirs = [os.path.join(data_dir, i) for i in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, i))]

    print("Speaker count: %s" % len(speakers_dirs))
    with Pool(processes=16) as pool:
        for speaker_dir in speakers_dirs:
            pool.apply_async(audio_clip_speaker, (speaker_dir, N, low, high, duration, output_dir))
        pool.close()
        pool.join()


def audio_clip_speaker(speaker_dir, N, low, high, duration, output_dir):
    speaker = os.path.basename(speaker_dir)
    print("Speaker:\t%s" % speaker)
    #print("Source:\t%s" % speaker_dir)

    audio_files = glob.glob(os.path.join(
        speaker_dir, "**/*.sph"), recursive=True)
    audio_files.extend(glob.glob(os.path.join(
        speaker_dir, "**/*.wav"), recursive=True))
    audio_files.extend(glob.glob(os.path.join(
        speaker_dir, "**/*.m4a"), recursive=True))

    #print("Source files:\t%d" % len(audio_files))

    p = os.path.join(output_dir, speaker)
    if not os.path.exists(p):
        os.makedirs(p)

    for j in range(N):
        audio_file = np.random.choice(audio_files)
        #print("\t%s" % audio_file)
        y, sampling_rate = sf.read(audio_file, dtype='float32')

        if len(y) > duration * sampling_rate:
            # Select randomly
            k = int(
                np.random.uniform(low,
                                  min(
                                      (len(y) - high*sampling_rate) / sampling_rate,
                                      (len(y) - duration*sampling_rate) / sampling_rate
                                  ), size=1))
            utterance = y[k *
                          sampling_rate: math.floor((k+duration)*sampling_rate)]
        else:
            # Use whole
            utterance = y

        librosa.output.write_wav(os.path.join(p, str(j)) + ".wav", utterance, sampling_rate)
