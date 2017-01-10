import time
import numpy as np
import theano
import theano.tensor as T
import lasagne

from astropy.io import fits

from .patch import extract_patches, augment, nanomaggie_to_luptitude
from .params import save_params, load_params
from .models import build_cnn


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


def load_training_set(filenames, bands, num_classes, size):

    image_data = []
    for f in filenames:
        image_data.append(fits.getdata(f))
    image_data = np.stack(image_data)
        
    X_train, y_train = extract_patches(
        image_data,
        size=size,
        num_patches=num_classes
    )

    for idx, band in enumerate(bands):
        X = nanomaggie_to_luptitude(X_train[:, idx, :, :], band)
        X_train[:, idx, :, :] = normalize(X)

    return X_train, y_train


def train_cnn(filenames, num_epochs=1000, num_classes=1000, size=16, bands=None, pretrained=None):

    print("Loading data...")

    if bands is None:
        bands = "ugriz"

    X_train, y_train = load_training_set(
        filenames=filenames,
        bands=bands,
        num_classes=num_classes,
        size=size
    )

    print("Training data loaded, shape: {}, min: {}, max: {}".format(X_train.shape, X_train.min(), X_train.max()))

    print("Compiling...")

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var, num_classes, size, len(filenames))

    prediction = lasagne.layers.get_output(network)
 
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
        dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)

    learning_rate = T.scalar("learning_rate")

    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate, 0.9
    )

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_loss = lasagne.objectives.categorical_crossentropy(
        test_prediction, target_var
    )
    test_loss = T.mean(test_loss, dtype=theano.config.floatX)

    test_acc = T.mean(
        T.eq(T.argmax(test_prediction, axis=1), target_var),
        dtype=theano.config.floatX
    )

    train_fn = theano.function(
        [input_var, target_var, learning_rate],
        loss,
        updates=updates,
        allow_input_downcast=True
    )

    val_fn = theano.function(
	inputs=[input_var, target_var],
        outputs=[test_loss, test_acc]
    )

    print("Starting training...")

    if pretrained:
        load_params(network, pretrained)

    batch_size = 128
    base_learning_rate = 0.001

    for epoch in range(num_epochs):

        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            inputs = augment(inputs)
            lr = base_learning_rate * np.power(1 + 0.01 * epoch, -1.0)
            train_err += train_fn(inputs, targets, lr)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        val_err_best = np.inf

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            if val_err < val_err_best:
                val_err_best = val_err
                save_params(network, "model.params")

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    return network
