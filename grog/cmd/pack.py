import argparse
from grog.data.audiopacker import PackData
from grog.config import Config
from grog.data.dataset_config import DatasetConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser("The function is to pack the audio files")
    parser.add_argument("-d", "--dir", type=str, help="root directory which contains the fold of audio files from each speaker", required=True)
    parser.add_argument("-o", "--out", type=str, help="output file name", required=True)
    parser.add_argument("--config", type=str, help="the config", required=True)
    args = parser.parse_args()

    config = DatasetConfig()


    config.load_json(args.config)

    gen = PackData(data_dir=args.dir, output=args.out, config=config)
    gen.pack()
