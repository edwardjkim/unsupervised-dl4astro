from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude
from sklearn.decomposition import PCA


def squares_of_gradient(array, size, sigma=1.0):
    """
    """

    result = np.zeros((array.shape[0], array.shape[1]))

    gradients = gaussian_gradient_magnitude(array, sigma=sigma)

    up = 0
    down = array.shape[0] - size
    left = 0
    right = array.shape[1] - size
    
    for i in range(up, down):
        for j in range(left, right):
            result[i, j] = np.sum(np.square(gradients[i: i + size, j: j + size]))

    return result


def compute_PCA(array):
    """
    Computes PCA.
    """

    nimages0, nchannels0, height0, width0 = array.shape
    rolled = np.transpose(array, (0, 2, 3, 1))
    # transpose from N x channels x height x width  to  N x height x width x channels
    nimages1, height1, width1, nchannels1 = rolled.shape
    # check shapes
    assert nimages0 == nimages1
    assert nchannels0 == nchannels1
    assert height0 == height1
    assert width0 == width1
    # flatten
    reshaped = rolled.reshape(nimages1 * height1 * width1, nchannels1)
    
    pca = PCA()
    pca.fit(reshaped)
    
    cov = pca.get_covariance()
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    return eigenvalues, eigenvectors


def augment(array):
    """
    """

    batch_size, nchannels, width, height = array.shape

    eigenvalues, eigenvectors = compute_PCA(array)

    # Flip half of the images horizontally at random
    indices = np.random.choice(batch_size, batch_size // 2, replace=False)        
    array[indices] = array[indices, :, :, ::-1]

    for i in range(batch_size):
        
        # Rotate 0, 90, 180, or 270 degrees at random
        nrotate = np.random.choice(4)
        
        # add random color perturbation
        alpha = np.random.normal(loc=0.0, scale=0.5, size=nchannels)
        noise = np.dot(eigenvectors, np.transpose(alpha * eigenvalues))
        
        for j in range(nchannels):
            array[i, j, :, :] = np.rot90(array[i, j, :, :] + noise[j], k=nrotate)

    return array


def nanomaggie_to_luptitude(array, band):
    """
    Converts nanomaggies (flux) to luptitudes (magnitude).
    http://www.sdss.org/dr12/algorithms/magnitudes/#asinh
    http://arxiv.org/abs/astro-ph/9903081
    """
    b = {
        'u': 1.4e-10,
        'g': 0.9e-10,
        'r': 1.2e-10,
        'i': 1.8e-10,
        'z': 7.4e-10
    }
    nanomaggie = array * 1.0e-9 # fluxes are in nanomaggies

    luptitude = -2.5 / np.log(10) * (np.arcsinh((nanomaggie / (2 * b[band]))) + np.log(b[band]))
    
    return luptitude


def extract_patches(array, size, num_patches, sigma=1.0, reference=None):
    """
    """

    nchannels, nrows, ncols = array.shape

    if reference is None:
        reference = nchannels // 2

    X = np.zeros((num_patches, nchannels, size, size))

    sog = squares_of_gradient(array[reference], size, sigma=sigma)

    sog = sog / np.sum(sog)

    # aligning images can produce NaN values in some pixels.
    # we will draw 10% more patches than specified and
    # skip images with any NaN values.
    nbuffer = int(0.1 * num_patches)

    pos = np.random.choice(
        np.arange(sog.size),
        size=num_patches + nbuffer,
        p=sog.reshape((sog.size, )),
        replace=False
    )

    row = pos // sog.shape[1]
    col = pos % sog.shape[1]

    for i in range(num_patches):
        icol = col[i]
        irow = row[i]
        patch = array[:, irow: irow + size, icol: icol + size]

        # as mentioned previously, some aligned images may have NaN values.
        # if there are any NaN values, continue.
        if np.sum(np.isnan(patch)) > 0:
            continue
        else:
            X[i] = patch

    X = X.astype(np.float32)
    y = np.arange(num_patches).astype(np.int32)

    return X, y
