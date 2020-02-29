#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

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

train 1g TEDLIUM/train.pkl TEDLIUM/evaluation.pkl TEDLIUM
train 1h TEDLIUM/train.pkl TEDLIUM/evaluation.pkl TEDLIUM
train 1i TEDLIUM/train.pkl TEDLIUM/evaluation.pkl TEDLIUM

train 3a WSJ0/train.pkl WSJ0/validation.pkl WSJ0
train 3b WSJ0/train.pkl WSJ0/validation.pkl WSJ0

train 4c WSJ0/train.pkl WSJ0/validation.pkl WSJ0
