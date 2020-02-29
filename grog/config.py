import json

WINDOWS_PER_SAMPLE = 100  # number of windows forming a chunk of data
SAMPLING_RATE = 8000
WINDOW_SIZE = 256
# TF bins smaller than THRESHOLD will be
# considered inactive
THRESHOLD = 40
# embedding dimention
EMBBEDDING_D = 40
# feed forward dropout prob
P_DROPOUT_FF = 0.5
# recurrent dropout prob
P_DROPOUT_RC = 0.2
N_HIDDEN = 600
N_LAYERS = 4
LAYER_NAME_OFFSET = 1
LEARNING_RATE = 1e-5
MAX_STEP = 2000000
TRAIN_BATCH_SIZE = 128


class Config(object):
    def __init__(self):
        pass

    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            config = json.load(file)
            self.load_dict(config)

    def loads_json(self, string):
        config = json.loads(string)
        self.load_dict(config)

    def load_dict(self, config):
        self.learning_rate = config['learning_rate']
        self.n_layers = config['n_layers']
        self.n_hidden = config['n_hidden']
        self.embedding_dimension = config['embedding_dimension']
        self.window_size = config['window_size']
        self.windows_per_sample = config['windows_per_sample']
        self.max_steps = config['max_steps']
        self.batch_size = config['batch_size']
        self.train_dropout_ff = config['train_dropout_ff']
        self.train_dropout_rc = config['train_dropout_rc']
        self.sampling_rate = config['sampling_rate']
        self.hop_length = config['hop_length']

        self.threshold = config['threshold']
        self.dataset_mean = config['dataset_mean']
        self.dataset_std = config['dataset_std']

        # TODO: Check nothing is None

    def print(self):
        print("Current configuration: %s" % json.dumps(self.__dict__, indent=4))
