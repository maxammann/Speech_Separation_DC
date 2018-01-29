import mir_eval
import numpy as np
import librosa
from infer import blind_source_separation

## source1 is the primary speaker
def audiomixer(source1, source2, frac=0.7):
  data1, _ = librosa.load(source1, sr=8000)
  data2, _ = librosa.load(source2, sr=8000)
  m = min(len(data1), len(data2))
  mix_data = data1[:m] + data2[:m] * frac
  mix_data = mix_data / max(abs(mix_data))
  source1 = source1.strip().split("/")
  source2 = source2.strip().split("/")
  speaker1, speaker2 = source1[-2], source2[-2]
  source1_name, source2_name = source1[-1][:-4], source2[-1][:-4]
  mix_name = "_".join([speaker1, source1_name, speaker2, source2_name, "mix.wav"])
  librosa.output.write_wav(mix_name, mix_data, 8000)

def evaluate(reference_sources, estimated_sources):
  m = len(reference_sources[0])
  for i in range(len(reference_sources)):
    m = min(m, len(reference_sources[i]))
    m = min(m, len(estimated_sources[i]))
  
  references = np.stack([source[:m] for source in reference_sources])
  estimates = np.stack([source[:m] for source in estimated_sources])
  sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources_framewise(
                         references, estimates)
  return sdr, sir, sar, perm

def test(source1, source2, frac=0.7):
  data1, _ = librosa.load(source1, sr=8000)
  data2, _ = librosa.load(source2, sr=8000)

  m = min(len(data1), len(data2))
  mix_data = data1[:m] + data2[:m] * frac
  mix_data = mix_data / max(abs(mix_data))
  source1 = source1.strip().split("/")
  source2 = source2.strip().split("/")
  speaker1, speaker2 = source1[-2], source2[-2]
  source1_name, source2_name = source1[-1][:-4], source2[-1][:-4]
  mix_name = "_".join([speaker1, source1_name, speaker2, source2_name, "mix.wav"])
  librosa.output.write_wav(mix_name, mix_data, 8000)

  sources = blind_source_separation(mix_name)

  for i in range(len(sources)):
    librosa.output.write_wav(mix_name[0:-7] + "_source" + str(i+1) + ".wav",
      sources[i][0], sources[i][1])

  reference_sources = [data1[:m], data2[:m]]
  estimated_sources = [sources[0][0], sources[1][0]]
  sdr, sir, sar, perm = evaluate(reference_sources, estimated_sources)
  return sdr, sir, sar, perm
