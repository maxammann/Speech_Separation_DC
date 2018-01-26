# A tensorflow implementation of deep clustering for blind source separation
This is a tensorflow implementation of the deep clustering paper: https://arxiv.org/abs/1508.04306
The code is based on zhr1201/deep-clustering in the Github.

## Requirements
  * Python 2.7
  * tensorflow
  * numpy
  * scikit-learn
  * matplotlib
  * librosa
  * mir_eval
  
## File documentation
  * audiopacker.py: pack audios from different speakers into .pkl format.
  * model.py: Bi-LSTM neural network.
  * train.py: train the deep learning network. 
  * infer.py: separate sources from the mixture.
  
## Training procedure
  * TODO

## References
  https://arxiv.org/abs/1508.04306
