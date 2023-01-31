import numpy as np
import math



def get_lower_triangular(array):
    """
    Returns a lower triangular matrix from an array which is conformable with such
    an operation (i.e. arrays whose length can be written as n(n+1)/2 for some positive integer
    n)

    :param array: (l x 1) Array of numbers
    :return: (n x n) lower triangular matrix where nonzero values come from 'array'
    """
    # Verify that the array is triangular
    l = len(array)
    n = (math.sqrt(1 + 8 * l) - 1) / 2
    assert (n == int(n))
    n = int(n)
    arr = []

    for i in range(n):
        a = np.zeros((1, n))
        a[0,:i+1] = array[int(i * (i + 1) / 2):int((i + 1) * (i + 2) / 2)]
        arr.append(a)

    if len(arr) == 0:
        return np.zeros((0,0))
    else:
        return np.concatenate(tuple(arr), axis=0)


def test_column_equal(array):
    """

    :param array:
    :return:
    """
    cols = array.shape[1]
    out = []
    for i in range(cols):
        print(np.max(array[:,i]) - np.min(array[:,i]))
        out.append(np.max(array[:,i]) == np.min(array[:,i]))

    return out

