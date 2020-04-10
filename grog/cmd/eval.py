import argparse
from grog.models.infer import Inference
from grog.config import Config
from grog.evaluation.evaluate import generate
from grog.evaluation.preparation import generate_mixtures
import hickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser("restore sound for each speaker")
    parser.add_argument("-i", "--eval_data_path", nargs="*", help="The path to evaluation utterances", required=True)
    parser.add_argument("-p", "--eval_pack_path", nargs="*", help="The pack cache", required=True)
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

    model_dir = args.model_dir
    eval_data_path = args.eval_data_path
    eval_pack_path = args.eval_pack_path
    output = args.output

    def generate_eval_data(n, sample_dir, out):
        if os.path.isfile(out):
            return hickle.load(out)

        sampling_rate = 8000
        print(out)
        generated_mixtures = generate_mixtures(sample_dir, sampling_rate, n)
        return generated_mixtures

    n = 100

    voxceleb = generate_eval_data(
        n, 
        eval_data_path, 
        eval_pack_path
    )

    eval_result = (config_path, config, "voxceleb", "voxceleb-set", generate(model_dir, config, voxceleb)) # TODO: Also store here evaluation results using eval_generated

    print("Dumping eval_result")
    hickle.dump(eval_result, output, compression='gzip')
