import numpy as np
import random
import timeit
import math

if __name__ == '__main__':

    lens = range(10, 300, 10)
    np_time = []

    for l in lens:
        rands = [random.random() for _ in range(0, l)]
        numpy_rands = np.array(rands)
        execution_time = timeit.timeit(lambda: np.std(numpy_rands), number=10000)
        np_time.append(execution_time)
        print("%d %f" % (l, execution_time))

    with open("data/time_per_array_size_numpy.dat", "w") as f:
        for i in range(len(np_time)):
            f.write("%d %f\n" % (lens[i], np_time[i]))
