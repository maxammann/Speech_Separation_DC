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
  
## Train your speaker separator
  * Put audio files in wav format under the data directory. For each speaker,
    one should create a folder and put all audios that belong to this speaker
    into this folder. As an example, one can download two audio files using the
    links as follows:
    https://drive.google.com/open?id=15zPtDBkb4VcxvgS7O3QOCqKdERMeEgEO
    https://drive.google.com/open?id=1m2b7reWlJ5qu5zh1cTFSBdOw9fU6jO9b. After
    downloading the files, one should put them into the data directory and type
    dewave-clip --dir=data --out=data/train --num=256 in the command line, which
    will automatically generate the training datasets. Similarly, one can type
    dewave-clip --dir=data --out=data/val --num=128 to obtain the validation
    data.
  * Pack the data. Type
    dewave-pack --dir=data/train --out=train.pkl
    dewave-pack --dir=data/val --out=val.pkl
    The train.pkl is used as the training data and the val.pkl is used as the
    validation data.
  * Train the model. In the python environment, import the package DeWave and 
    type DeWave.train.train()
  * Stop the training process once the loss on the validation datasets
    converges.
  * For a mixed audio file, e.g. mix.wav, type
    sources = DeWave.infer.blind_source_separation("mix.wav")
    to retrivial the sources and use 
    librosa.output.write_wav("source1.wav", sources[0][0], sources[0][1])
    librosa.output.write_wav("source2.wav", sources[1][0], sources[1][1])
    to write two estimated audios into files.

## Pretrained model
  I have a pretrained model using TED talks from 5 speakers. One can download
  the model through the link below:
  https://drive.google.com/open?id=1MMOMKlNI0-wIRUrNIX4KlFRrkm6UvSi1. 
  Put the checkpoint and other seed files under the seeds directory. Then one
  can decompose mixed audios using DeWave.infer.blind_source_separation.

## References
  https://arxiv.org/abs/1508.04306

## Troubleshooting
1. Error for reading the audio file using librosa.
   Solution: install ffmpeg.
