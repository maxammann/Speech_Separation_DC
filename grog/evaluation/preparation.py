import os
import glob
import numpy as np
import librosa
import soundfile as sf

def generate_mixtures(data_dir, sampling_rate, n):
    speakers_dirs = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))]
    speakers_dirs.sort()
    print("# of mixes: %d" % n)
    print("\t with sampling rate: %d" % sampling_rate)
    print("\t out of %d speakers" % len(speakers_dirs))

    mixes = []
    reference = []
    labels = []

    def augment(speech):
        fac = np.random.uniform(-3, 3)
        return 10. ** (fac / 20) * speech

    def get_speaker_audios(speaker_dir):
        audio_files = glob.glob(os.path.join(
            speaker_dir, "**/*.wav"), recursive=True)
        return sorted(audio_files)

    for j in range(n):
        speaker_dir = np.random.choice(speakers_dirs)
        other_speaker_dir = np.random.choice(speakers_dirs)

        while(speaker_dir == other_speaker_dir):
            other_speaker_dir = np.random.choice(speakers_dirs)

        audio_file = np.random.choice(get_speaker_audios(speaker_dir))
        other_audio_file = np.random.choice(get_speaker_audios(other_speaker_dir))
                
        y1, sampling_rate1 = sf.read(audio_file, dtype='float32')
        y2, sampling_rate2 = sf.read(other_audio_file, dtype='float32')

        if sampling_rate1 != sampling_rate or sampling_rate2 != sampling_rate:
            raise Exception('Invalid sampling rate')

        length = min(len(y1), len(y2))
        #y1 = augment(y1[:length])
        #y2 = augment(y2[:length])
        y1 = y1[:length]
        y2 = y2[:length]
        mix = y1 + y2
        
        speaker_names = (os.path.basename(speaker_dir), os.path.basename(other_speaker_dir))
        
        reference.append([y1, y2])
        mixes.append(mix)

        labels.append(speaker_names)
    return mixes, reference, labels