import sys
import numpy as np


def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    print(b)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    print(shp)
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


if __name__ == "__main__":

    data = np.load("../p1_data/p1_skip5_10ns.npy")
    shp = data.shape[0]
    for ndx in range(shp):
        np.random.shuffle(data[ndx, :, :])

    np.save("../p1_data/p1_skip5_10ns_shf.npy", data)
