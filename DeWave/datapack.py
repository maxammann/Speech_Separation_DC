#!/usr/bin/env python

## Script is used to pack audio clips
import numpy as np
import librosa
import pickle
from numpy.lib import stride_tricks
import os
from .constant import *
import argparse
import glob
from .audiopacker import PackData


parser = argparse.ArgumentParser("The function is to pack the audio files")
parser.add_argument("-d", "--dir", type=str, help="root directory which \
                    contains the fold of audio files from each speaker")
parser.add_argument("-o", "--out", type=str, help="output file name")
args = parser.parse_args()

def packclips():
    gen = PackData(data_dir=args.dir, output=args.out)
    gen.reinit()
