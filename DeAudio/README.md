# Single-channel blind source separation
This package decomposes two overlapping speech signals, which are recoded in one channel. 
The method is described in deep clustering paper: https://arxiv.org/abs/1508.04306.
The code is based on [zhr1201/deep-clustering](https://github.com/zhr1201/deep-clustering) from the Github.

## Requirements
  * Python3
  * tensorflow
  * numpy
  * scikit-learn
  * librosa
  * mir_eval
  
## File documentation
  * traindata.py: generate audio clips for training and validation
  * audiopacker.py: pack audios from different speakers into .pkl format.
  * model.py: Bi-LSTM neural network.
  * train.py: train the deep learning network. 
  * infer.py: separate sources from the mixture.
  * utility.py: evaluate the performance of the DNN model
  
## Training procedure
  * TODO

## References
  https://arxiv.org/abs/1508.04306
