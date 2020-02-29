#!/bin/bash

function train {
    experiment=$1
    train=$2
    val=$3
    dataset=$4
    echo Experiment: $experiment - $dataset
    python3 -u -m grog.cmd.train \
        --model_dir ../exp-models/${experiment}/seeds \
        --summary_dir ../exp-models/${experiment}/summary \
        --train_pkl ../nas/data/${train} \
        --val_pkl ../nas/data/${val} \
        --config experiments/${experiment}/config.json 2>&1 | tee $experiment.$dataset.log
}

#SBATCH --partition=dfl
#SBATCH --time=2:00:00
SBATCH --mem=30000
#SBATCH --gres=gpu:1
SBATCH --ntasks=1
#SBATCH --get-user-env
#SBATCH --export=ALL
SBATCH -o slurm.out
SBATCH -J dc

train 4b WSJ0/train.pkl WSJ0/validation.pkl WSJ0