import numpy as np
import random
import timeit
import math
import sys

if __name__ == '__main__':

    numpy_version = np.__version__.split('.')[1]
    lens = range(10, 300, 10)
    np_time = []

    for l in lens:
        rands = [random.random() for _ in range(0, l)]
        numpy_rands = np.array(rands)
        execution_time = timeit.timeit(lambda: np.std(numpy_rands), number=10000)
        np_time.append(execution_time)
        print("array_size: %d, execution_time:%f, numpy_version:%s" % (l, execution_time, numpy_version))

    with open("data/time_per_array_size_numpy_%s.dat" % numpy_version, "w") as f:
        for i in range(len(np_time)):
            f.write("%d %f\n" % (lens[i], np_time[i]))
