import numpy as np
import math


def get_lower_triangular(array):
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
