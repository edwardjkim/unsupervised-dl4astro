import lasagne
from lasagne.layers import (
    InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer, DropoutLayer,
    batch_norm
)
from lasagne.nonlinearities import rectify, softmax
from lasagne.init import HeNormal


def build_cnn(input_var, num_outputs, size, num_channels=3):

    network = InputLayer(shape=(None, num_channels, size, size), input_var=input_var)

    network = batch_norm(Conv2DLayer(
        network,
        num_filters=64, filter_size=(5, 5),
        nonlinearity=rectify, W=HeNormal()
    ))
    network = MaxPool2DLayer(network, pool_size=(3, 3), stride=1)
    network = DropoutLayer(network, p=0.5)

    network = batch_norm(Conv2DLayer(
        network,
        num_filters=128, filter_size=(5, 5),
        nonlinearity=rectify, W=HeNormal()
    ))
    network = MaxPool2DLayer(network, pool_size=(3, 3), stride=1)
    network = DropoutLayer(network, p=0.5)

    network = DenseLayer(
        network,
        num_units=num_outputs, nonlinearity=softmax,
        W=HeNormal()
    )

    return network

