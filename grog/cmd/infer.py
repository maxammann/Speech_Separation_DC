import argparse
from grog.models.infer import Inference
from grog.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("restore sound for each speaker")
    parser.add_argument("-i", "--input_files", nargs="*", help="the mixed audio files", required=True)
    parser.add_argument("-m", "--model_dir", type=str, help="the directory where the trained model is stored", required=True)
    parser.add_argument("-o", "--output", type=str, help="the directory where the estimated sources should be stored", required=True)
    parser.add_argument("--config", type=str, help="the config", required=False)
    args = parser.parse_args()

    config = Config()

    if args.config:
        config.load_json(args.config)
    else:
        config.load_constants()

    config.print()

    Inference(config).blind_source_separation(args.input_files, args.model_dir, args.output)
