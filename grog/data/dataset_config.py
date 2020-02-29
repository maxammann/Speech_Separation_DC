import json

class DatasetConfig(object):
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
        self.duration = config['duration']
        self.windows_per_sample = config['windows_per_sample']
        self.sampling_rate = config['sampling_rate']
        self.window_size = config['window_size']
        self.hop_length = config['hop_length']
        self.threshold = config['threshold']
        self.augmentation_factor_min = config['augmentation_factor_min']
        self.augmentation_factor_max = config['augmentation_factor_max']

    def print(self):
        print("Current configuration: %s" % json.dumps(self.__dict__, indent=4))
