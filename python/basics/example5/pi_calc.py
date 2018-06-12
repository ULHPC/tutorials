"""
Calculation of Pi using a Monte Carlo method.
"""

from math import hypot
from random import random
from scoop import futures
from time import time
import sys
import filelock

# A range is used in this function for python3. If you are using python2, a
# xrange might be more efficient.
def test(tries):
    return sum(hypot(random(), random()) < 1 for _ in range(tries))


# Calculates pi with a Monte-Carlo method. This function calls the function
# test "n" times with an argument of "t". Scoop dispatches these 
# functions interactively accross the available ressources.
def calcPi(workers, tries):
    bt = time()
    expr = futures.map(test, [tries] * workers)
    piValue = 4. * sum(expr) / float(workers * tries)
    totalTime = time() - bt
    print("pi = " + str(piValue))
    print("total time: " + str(totalTime))
    return (piValue, totalTime)

if __name__ == "__main__":
    dataPi, totalTime = calcPi(3000, 50000)
    with filelock.FileLock("scoop.lock") as lock:
        with open("data/time_per_core.dat", "a") as f:
            # nbcore time
            f.write("%d %f\n" % (int(sys.argv[1]), totalTime))
