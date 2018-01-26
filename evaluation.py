import mir_eval
import numpy as np
import librosa

def evaluate(reference_sources, estimated_sources):
  m = 0
  for i in range(len(reference_sources)):
    m = min(m, len(reference_sources[i]))
    m = min(m, len(estimated_sources[i]))
  
  references = np.stack([source[:m] for source in reference_sources])
  estimates = np.stack([source[:m] for source in estimated_sources])
  sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources_framewise(sources, 
                                                                       estimates)
  return sdr, sir, sar, perm
