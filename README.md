# Single-channel blind source separation
This package decomposes two overlapping speech signals, which are recoded in one channel. 
The method is described in deep clustering paper: https://arxiv.org/abs/1508.04306.
The code is based on [zhr1201/deep-clustering](https://github.com/zhr1201/deep-clustering) from the Github.
I fixed the issue of inference using the trained model, upgraded the code to
support python3 and made a python package called DeWave.

## Requirements
  * Python3.6
  * tensorflow
  * numpy
  * scikit-learn
  * librosa
  * mir_eval

## Installation
The python is available on PyPI, and you can install it by typing
`pip install DeWave`
  
## File documentation
  * model.py: Bi-LSTM neural network
  * train.py: train the deep learning network
  * infer.py: separate sources from the mixture
  * utility.py: evaluate the performance of the DNN model
  
## Training procedure
  * TODO

## References
  https://arxiv.org/abs/1508.04306

## Troubleshooting
1. Error for reading the audio file using librosa.
   Solution: install ffmpeg.
