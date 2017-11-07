import random
import timeit
import math
import sys
import std

if __name__ == '__main__':

    lens = range(10, 300, 10)
    c_time = []

    for l in lens:
        rands = [random.random() for _ in range(0, l)]
        execution_time = timeit.timeit(lambda: std.standard_dev(rands), number=10000)
        c_time.append(execution_time)
        print("array_size: %3d, execution_time:%f" % (l, execution_time))

    with open("data/time_per_array_size_c.dat", "w") as f:
        for i in range(len(c_time)):
            f.write("%d %f\n" % (lens[i], c_time[i]))
