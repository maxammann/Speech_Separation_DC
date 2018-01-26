import argparse
import inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser("blind source separation")
    parser.add_argument("-i", "--dir", type=str, help="input folder name")
    args = parser.parse_args()
    data_dir=args.dir
    ## decompose audios with the suffix "mix" at the end
    files = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if i[-7:-4] == "mix"]
    for i in files:
      inference.blind_source_separation(i)
