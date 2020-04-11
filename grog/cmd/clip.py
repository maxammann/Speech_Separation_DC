
from grog.data.clip import clip
import argparse
from grog.config import Config

# Script is used to generate training and validation datasets from sph files
if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate training data")
    parser.add_argument("--dir", type=str, help="input folder name", required=True)
    parser.add_argument("--num", type=int, help="The number of clips for each speaker", default=128)
    parser.add_argument("--duration", type=float, help="The duration of each clip", default=5)
    parser.add_argument("--low", type=float, help="starting time of the audio from which the clip is sampled", default=0)
    parser.add_argument("--high", type=float, help="(sample count - high) is the end of the audio from which the clip is sampled", default=600)
    parser.add_argument("--out", type=str, help="the output directory", required=True)
    parser.add_argument("--config", type=str, help="the config", required=False)
    args = parser.parse_args()

    config = Config()

    if args.config:
        config.load_json(args.config)
    else:
        config.load_constants()

    config.print()

    clip(args.dir, args.num, args.low, args.high, args.duration, args.out, config)
