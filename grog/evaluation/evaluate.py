import os
import glob
import sys
import itertools

import numpy as np
import tensorflow as tf
import librosa

from grog.config import Config
from grog.models.infer import Inference
import IPython.display as ipd
from grog.fft import stft_default

from museval.metrics import bss_eval
from grog.util import pad_or_truncate

def eval_estimated(reference, ref_labels, estimated, est_labels, sampling_rate, win, hop, mode, compute_permutation=True):
    if len(reference) != len(estimated):
        raise Exception("You need the same amount of references as estimates!")

    reference_np, estimated_np = pad_or_truncate(reference, estimated)

    SDR, ISR, SIR, SAR, _ = bss_eval(
        reference_np,
        estimated_np,
        compute_permutation=compute_permutation,
        window=int(win*sampling_rate),
        hop=int(hop*sampling_rate),
        framewise_filters=(mode == "v3"),
        bsseval_sources_version=False
    )

    source_results = []
    for i, (ref_label, est_labels) in enumerate(zip(ref_labels, est_labels)):
        source_results.append({
            "reference_name": ref_label,
            "estimated_name": est_labels,
            "SDR": SDR[i].tolist(),
            "SIR": SIR[i].tolist(),
            "ISR": ISR[i].tolist(),
            "SAR": SAR[i].tolist()
        })
    return source_results

def eval_generated(model_dir, config, generated_mixtures, window_size=1.0, hop_length=1.0, mode='v4', debug=False):
    sampling_rate = config.sampling_rate
    mixes, reference, labels = generated_mixtures
    eval_results = []
    ret_sources = []

    inference = Inference(config)
    
    session, embedding_model, in_data, in_data, dropout_ff, dropout_rc = inference.prepare_session(model_dir, debug)
    
    for mix, ref, label in zip(mixes, reference, labels):
        embeddings, N_samples = inference.estimate_embeddings(session, mix, embedding_model, in_data, dropout_ff, dropout_rc)
        sources = inference.estimate_sources(mix, embeddings, N_samples, ref)

        est_labels = range(2)
        result = eval_estimated(ref, label, sources, est_labels, sampling_rate, window_size, hop_length, mode)
        #result_rev = eval_estimated(ref, label, list(reversed(sources)), list(reversed(est_labels)), sampling_rate, window_size, hop_length, mode)
        result_rev = None
        result_baseline = eval_estimated(ref, labels, [mix] * 2, ["mixture"] * 2, sampling_rate, window_size, hop_length, mode)
        #result_baseline = None

        eval_results.append((result, result_rev, result_baseline))
        ret_sources.append(sources)


    #session.close()

    return eval_results, mixes, reference, labels, ret_sources