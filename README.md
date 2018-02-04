# Single-channel blind source separation
This package decomposes two overlapping speech signals, which are recoded in one channel. 
The method is described in deep clustering paper: https://arxiv.org/abs/1508.04306.
The code is based on [zhr1201/deep-clustering](https://github.com/zhr1201/deep-clustering) from the Github.
I fixed issues with inference using the trained model, upgraded the code to
support python3 and made a python package called DeWave.

## Requirements
  * Python3.6
  * tensorflow
  * numpy
  * scikit-learn
  * librosa

## Installation
The python is available on PyPI, and you can install it by typing
`pip install DeWave`
  
## File documentation
  * model.py: Bi-LSTM neural network
  * train.py: train the deep learning network
  * infer.py: separate sources from the mixture
  * utility.py: evaluate the performance of the DNN model
  
## Training your speaker separator
### Prepare training and validation datasets
  * Put audio files in wav or sph format under the data directory. For each speaker,
    one should create a folder and put all audios that belong to this speaker
    into this folder. The function `dewave-clip` can help generate clips based
    on these audios. As an example, one can download two audio files using the
    links as follows:  
    https://drive.google.com/open?id=15zPtDBkb4VcxvgS7O3QOCqKdERMeEgEO  
    https://drive.google.com/open?id=1m2b7reWlJ5qu5zh1cTFSBdOw9fU6jO9b.  
    After downloading the files and put them into the data directory. Under the
    current working directory, create a directory called data. Then type  
    `dewave-clip --dir=data --out=data/train --num=256`  
    in the command line, which will automatically generate the training datasets.
    Similarly, one can type  
    `dewave-clip --dir=data --out=data/val --num=128`  
    to obtain the validation data.
  * Pack the data. Type  
    `dewave-pack --dir=data/train --out=train.pkl`  
    `dewave-pack --dir=data/val --out=val.pkl`  
    The train.pkl is used as the training data and the val.pkl is used as the
    validation data.
### Train the DNN
  * Create two directories. One is used to store trained
    model. The other directory is used to store summary of learning process.
    For example, under the current working directory, we create two directoies,
    namely seeds and summary. Then one can type   
    `dewave-train --model_dir=seeds --summary_dir=summary --train_pkl=train.pkl --val_pkl=val.pkl`  
    in commmand line to start training the DNN model.
  * Stop the training process once the loss on the validation datasets
    converges.
## Infering based on trained model
  * For a mixed audio file, e.g. mix.wav, type
    `dewave-infer --input_file=mix.wav --model_dir=seeds`  
    in command line to restore the sources. Two restored audios called mix_source1.wav and 
    mix_source2.wav are generated. 

## Pretrained model
  I have a pretrained model using TED talks from 5 speakers. One can download
  the model through the link below:  
  https://drive.google.com/open?id=1MMOMKlNI0-wIRUrNIX4KlFRrkm6UvSi1. 

## References
  https://arxiv.org/abs/1508.04306

## Troubleshooting
  1. Error for reading the audio file using librosa.
     Solution: install ffmpeg.

  2. ValueError: Cannot feed value of shape (X, 100, 129) for Tensor
     'Placeholder_2:0', which has shape '(128, 100, 129)'. The number of audio
     clips should be at least 128. 
