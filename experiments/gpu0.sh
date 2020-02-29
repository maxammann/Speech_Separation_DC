#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

function train {
    experiment=$1
    train=$2
    val=$3
    dataset=$4
    echo Experiment: $experiment - $dataset
    python3 -u -m grog.cmd.train \
        --model_dir exp-models/${experiment}/seeds \
        --summary_dir exp-models/${experiment}/summary \
        --train_pkl data/${train} \
        --val_pkl data/${val} \
        --config ../sync/experiments/${experiment}/config.json 2>&1 | tee $experiment.$dataset.log
}

train 0 WSJ0/train.pkl WSJ0/validation.pkl WSJ0
train 1a WSJ0/train.pkl WSJ0/validation.pkl WSJ0
train 1c WSJ0/train.pkl WSJ0/validation.pkl WSJ0

train 2a WSJ0/train.pkl WSJ0/validation.pkl WSJ0
train 2b WSJ0/train.pkl WSJ0/validation.pkl

train 4a WSJ0/train.pkl WSJ0/validation.pkl WSJ0
