#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

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

train 1d TIMIT/train.pkl TIMIT/evaluation.pkl TIMIT
train 1e TIMIT/train.pkl TIMIT/evaluation.pkl TIMIT
train 1f TIMIT/train.pkl TIMIT/evaluation.pkl TIMIT

train 2d WSJ0/train.pkl WSJ0/validation.pkl WSJ0

train 4b WSJ0/train.pkl WSJ0/validation.pkl WSJ0
