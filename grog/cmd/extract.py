import argparse
from grog.models.infer import Inference
from grog.config import Config
import hickle
import librosa
import os

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
    output = args.output

    print("Loading eval_result")
    # eval_result = (config_path, config, "voxceleb", set_name, eval_generated(model_dir, config, voxceleb))
    config_path, config, name, set_name, eval_data  = hickle.load(eval_result)

    print("config_path:\t%s" % config_path)
    print("config:\t%s" % config)
    print("name:\t%s" % name)
    print("set_name:\t%s" % set_name)

    metrics, mixes, reference, labels, sources = eval_data

    print(len(mixes))

    for i, mix in enumerate(mixes):
        librosa.output.write_wav(os.path.join(output, "%dmix.wav") % i, mix, config.sampling_rate)
        librosa.output.write_wav(os.path.join(output, "%dsource1.wav") % i, sources[i][0], config.sampling_rate)
        librosa.output.write_wav(os.path.join(output, "%02dsource2.wav") % i, sources[i][1], config.sampling_rate)
    
