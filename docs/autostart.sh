#!/bin/bash

while [ ! -f 2019-05-12/train.pkl ]
do
          sleep 2
done
ls -la 2019-05-12/train.pkl >> check.txt
sleep 600
ls -la 2019-05-12/train.pkl >> check.txt
sleep 1
ls -la 2019-05-12/train.pkl >> check.txt
python3 ../DeWave/cli-train.py --model_dir=2019-05-12/model/seeds --summary_dir=2019-05-12/model/summary --train_pkl=2019-05-12/train.pkl --val_pkl=2019-05-12/val.pkl

