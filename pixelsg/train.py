import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (
    InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer,
    dropout, batch_norm
)
from lasagne.nonlinearities import rectify, softmax
from lasagne.init import GlorotUniform

from astropy.io import fits

from .patch import extract_patches, augment


def build_cnn(input_var, num_outputs):

    network = InputLayer(shape=(None, 1, 17, 17), input_var=input_var)

    network = batch_norm(Conv2DLayer(
        dropout(network, p=0.1),
        num_filters=32, filter_size=(5, 5),
        nonlinearity=rectify, W=GlorotUniform()
    ))

    network = MaxPool2DLayer(network, pool_size=(3, 3), stride=1)

    network = batch_norm(Conv2DLayer(
        dropout(network, p=0.2),
        num_filters=64, filter_size=(5, 5),
        nonlinearity=rectify
    ))

    network = MaxPool2DLayer(network, pool_size=(3, 3), stride=1)

    network = batch_norm(Conv2DLayer(
        dropout(network, p=0.3),
        num_filters=128, filter_size=(5, 5),
        nonlinearity=rectify
    ))

    network = DenseLayer(
        dropout(network, p=0.5),
        num_units=256, nonlinearity=rectify
    )

    network = DenseLayer(
        dropout(network, p=0.5),
        num_units=num_outputs, nonlinearity=softmax
    )

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def normalize(x):
    result = x - x.mean()
    result = (x - x.min()) / (x.max() - x.min())
    return result


def train_cnn(num_epochs=500):

    print("Loading data...")

    n_classes = 1000

    image_data = fits.getdata('/home/jovyan/work/shared/image-cutout/frame-r-000109-2-0037.fits')
    X_train, y_train = extract_patches(image_data, 17, n_classes)
    X_train = normalize(X_train)

    print("shape: {}, min: {}, max: {}".format(X_train.shape, X_train.min(), X_train.max()))

    print("Compiling...")

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var, n_classes)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
        dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9
    )

    def single_output(input_var):
        net = build_cnn(input_var, n_classes)
        result = lasagne.layers.get_output(net)
        return result

    # we use scan to do multiple forward passes
    # by using dropout and aggregating the results, we can estimate uncertainty
    # https://arxiv.org/abs/1506.02158
    scan_results, scan_updates = theano.scan(
        fn=lambda inputs_var: lasagne.layers.get_output(
            build_cnn(input_var, n_classes)
        ),
        n_steps=10,
        non_sequences=[input_var]
    )
    test_prediction = T.mean(scan_results, axis=0)
 
    test_loss = lasagne.objectives.categorical_crossentropy(
        test_prediction, target_var
    )
    test_loss = T.mean(test_loss, dtype=theano.config.floatX)

    test_acc = T.mean(
        T.eq(T.argmax(test_prediction, axis=1), target_var),
        dtype=theano.config.floatX
    )

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function(
        inputs=[input_var, target_var],
        outputs=[test_loss, test_acc],
        updates=scan_updates
    )

    print("Starting training...")

    batch_size = 128
    for epoch in range(num_epochs):

        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            inputs = augment(inputs)
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    return network

if __name__ == "__main__":

    _ = train_cnn(num_epochs=2000)
