import argparse
from grog.models.train import train
from grog.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("The function is to train the model")
    parser.add_argument("-m", "--model_dir", type=str, help="the directory where the trained model is stored", required=True)
    parser.add_argument("-s", "--summary_dir", type=str, help="the directory where the summary is stored", required=True)
    parser.add_argument("--train_pkl", type=str, nargs='+', help="file name of the training data", required=True)
    parser.add_argument("--val_pkl", type=str, nargs='+', help="file name of the validation data", required=True)
    parser.add_argument("--config", type=str, help="the config", required=True)
    args = parser.parse_args()

    config = Config()
    config.load_json(args.config)

    train(args.model_dir, args.summary_dir, args.train_pkl, args.val_pkl, config)
