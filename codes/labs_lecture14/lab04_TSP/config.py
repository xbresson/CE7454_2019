import json


class Settings(dict):
    """Experiment configuration options.

    Wrapper around in-built dict class to access members through the dot operation.

    Experiment parameters:
        "expt_name": Description of experiment, used for logging.
        "gpu_id": Available GPU ID(s)
        "train_filepath": Training set path
        "val_filepath": Validation set path
        "test_filepath": Test set path
        "num_nodes": Number of nodes in TSP tours
        "node_dim": Number of dimensions for each node
        "voc_nodes_in": Input node signal vocabulary size
        "voc_edges": Edge signal vocabulary size
        "beam_size": Beam size for beamsearch procedure (-1 for disabling beamsearch)
        "hidden_dim": Dimension of GCN hidden state
        "num_layers": Number of layers in GCN
        "max_epochs": Maximum training epochs
        "batch_size": Batch size
        "batches_per_epoch": Batches per epoch (-1 for using full set)
        "accumulation_steps": Number of steps for gradient accumulation
        "learning_rate": Initial learning rate
        "decay_rate": Rate of learning rate decay
    """

    def __init__(self, config_dict):
        super().__init__()
        for key in config_dict:
            self[key] = config_dict[key]

    def __getattr__(self, attr):
        return self[attr]

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    __delattr__ = dict.__delitem__


def get_default_config():
    """Returns default settings object.
    """
    return Settings(json.load(open("./configs/default.json")))


def get_config(filepath):
    """Returns settings from json file.
    """
    config = get_default_config()
    config.update(Settings(json.load(open(filepath))))
    return config
