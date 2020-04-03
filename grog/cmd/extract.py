import argparse
from grog.models.infer import Inference
from grog.config import Config
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unpack eval_results from eval step")
    parser.add_argument("-e", "--eval_result", type=str, help="Output file from eval command", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output dir for mixtures, references and estimated files", required=True)
    parser.add_argument("--config", type=str, help="the config", required=False)
    args = parser.parse_args()

    config = Config()

    if args.config:
        config.load_json(args.config)
    else:
        config.load_constants()

    config.print()

    eval_result = args.eval_result

    print("Loading eval_result")
    # eval_result = (config_path, config, "voxceleb", set_name, eval_generated(model_dir, config, voxceleb))
    config_path, config, name, set_name, eval_data  = pickle.load(open(eval_result, "rb"))

    print("config_path:\t%s" % config_path)
    print("config:\t%s" % config)
    print("name:\t%s" % name)
    print("set_name:\t%s" % set_name)

    metrics, mixes, reference, labels, sources = eval_data

    print(len(mixes))
    
