"""
Calculation of Pi using a Monte Carlo method.
"""

from math import hypot, pi
from random import random
from time import time
import sys
import filelock

# A range is used in this function for python3. If you are using python2, a
# xrange might be more efficient.
def test(tries):
    return sum(hypot(random(), random()) < 1 for _ in range(tries))


# Calculates pi with a Monte-Carlo method. This function calls the function
# test "n" times with an argument of "t". 
def calcPi(workers, tries):
    bt = time()
    expr = map(test, [tries] * workers)
    piValue = 4. * sum(expr) / float(workers * tries)
    totalTime = time() - bt
    print("pi = " + str(piValue))
    print("total time: " + str(totalTime))
    return (piValue, totalTime)

if __name__ == "__main__":
    tries = int(sys.argv[1])
    dataPi, totalTime = calcPi(tries, 1)
    with filelock.FileLock("pi_calc.lock") as lock:
        with open("data/fitness_per_tries.dat", "a") as f:
            # tries pivalue time
            f.write("%d %f\n" % (tries, (1-abs(dataPi-pi)/pi)*100))
        with open("data/time_per_tries.dat", "a") as f:
            # tries pivalue time
            f.write("%d %f\n" % (tries, totalTime))
