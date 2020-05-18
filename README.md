speech-separation-thesis
==============================

See final presentation [here](https://maxammann.github.io/speech-separation-presentation/index.html)



# Setup

* Install Anaconda 3
* Install Tenforflow from Anaconda: `conda install -c conda-forge tensorflow=1.13.1`
* `pip install -e .`

# Example usage

## Training

* `python3 -m grog.cmd.clip --dir continuous-train --out train --num=100 --low=20 --high=20 --duration=5`
* `python3 -m grog.cmd.pack --dir train --out train.pkl --config train.json`
* `CUDA_VISIBLE_DEVICES=1 python3 -m grog.cmd.train --model_dir models/wsj0-2_no-amp-fac/seeds --summary_dir models/wsj0-2_no-amp-fac/summary --train_pkl data/WSJ0/train.pkl --val_pkl data/WSJ0/validation.pkl --config models/wsj0-2_no-amp-fac/config.json`

## Inferring

* `python3 -m grog.cmd.clip --dir continuous-evaluation --out evaluation --num=200 --low=20 --high=20 --duration=5`
* `python3 -m grog.cmd.pack --dir evaluation --out evaluation.pkl --config evaluation.json`
* `CUDA_VISIBLE_DEVICES=0 python3 -m grog.cmd.infer --input_files workspace/eval/2019-05-25/FADG0-0_MABW0-0.wav --output workspace/eval/2019-05-25/model-novad --model_dir workspace/models/2019-05-25/model-novad/seeds`

# Evaluating

* `python3 -m grog.cmd.eval --config /fast/maxammann/config.json --eval_data_path /rzhome/ammannma/test-clip --eval_pack_path /fast/maxammann/eval_pack.pkl --model_dir /fast/maxammann/11_04/models/voxceleb  --output /fast/maxammann/eval_result.pkl`
* `python3 -m grog.cmd.extract --config /fast/maxammann/config.json --eval_result /rzhome/ammannma/eval/eval_result.pkl --output /rzhome/ammannma/eval/eval_result/`