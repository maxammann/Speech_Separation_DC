import numpy as np

def pad_or_truncate(audio_reference, audio_estimates):
    new_audio_reference = []
    
    for audio_est, audio_ref in zip(audio_estimates, audio_reference):
        est_shape = audio_est.shape
        ref_shape = audio_ref.shape
        if est_shape != ref_shape:
            if est_shape <= ref_shape:
                audio_ref = audio_ref[:est_shape[0]]
            else:
                # Padding can happen when the inferred audio sources are longer than the mix
                audio_ref = np.pad(
                    audio_ref,
                    [
                        (0, est_shape[0] - ref_shape[0])
                    ],
                    mode='constant'
                )
        new_audio_reference.append(audio_ref)
    return np.array(new_audio_reference), np.array(audio_estimates)
