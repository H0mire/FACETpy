import numpy as np


def crosscorrelation(x, y, maxlag, mode="corr"):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    # ensure that x an y are the same length
    len_diff = len(x) - len(y)

    if len_diff > 0:
        # x ist länger, also polstere y nur am Ende auf
        y = np.pad(y, (0, len_diff), mode="constant")
    elif len_diff < 0:
        # y ist länger, also polstere x nur am Ende auf
        x = np.pad(x, (0, abs(len_diff)), mode="constant")

    py = np.pad(y.conj(), 2 * maxlag, mode="constant")
    T = np.lib.stride_tricks.as_strided(
        py[2 * maxlag :],
        shape=(2 * maxlag + 1, len(y) + 2 * maxlag),
        strides=(-py.strides[0], py.strides[0]),
    )
    px = np.pad(x, maxlag, mode="constant")
    if mode == "dot":  # get lagged dot product
        return T.dot(px)
    elif mode == "corr":  # gets Pearson correlation
        return (T.dot(px) / px.size - (T.mean(axis=1) * px.mean())) / (
            np.std(T, axis=1) * np.std(px)
        )
