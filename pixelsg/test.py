import numpy as np
import theano
import theano.tensor as T
import lasagne


def test_cnn(network, num_passes=50):

    input_var = T.tensor4('inputs')

    # we use scan to do multiple forward passes
    # by using dropout and aggregating the results, we can estimate uncertainty
    # https://arxiv.org/abs/1506.02158
    num_passes = T.iscalar("num_passes")

    scan_results, scan_updates = theano.scan(
        fn=lambda: lasagne.layers.get_output(network),
        n_steps=num_passes
    )
    test_prediction = T.mean(scan_results, axis=0)

    test_fn = theano.function([input_var], [test_prediction])

