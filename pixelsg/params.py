from six.moves import cPickle as pickle
import os

import lasagne


def load_params(model, filename):
    """
    Unpickles and loads parameters into a Lasagne model.
    """
    filename = os.path.join(os.getcwd(), filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def save_params(model, filename):
    """
    Pickle the parameters of a Lasagne model.
    """
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join(os.getcwd(), filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
