"""
Calculation of Pi using a Monte Carlo method.
"""

from math import hypot
from random import random
from time import time
import sys
import numpy as np

# A range is used in this function for python3. If you are using python2, a
# xrange might be more efficient.
def test(tries):
    f = lambda x: hypot(random(), random()) < 1
    vf = np.vectorize(f)
    results = np.zeros(tries)
    return vf(results).sum()


# Calculates pi with a Monte-Carlo method. This function calls the function
# test "n" times with an argument of "t". Scoop dispatches these 
# functions interactively accross the available ressources.
def calcPi(workers, tries):
    bt = time()
    expr = map(test, [tries] * workers)
    piValue = 4. * sum(expr) / float(workers * tries)
    totalTime = time() - bt
    print("pi = " + str(piValue))
    print("total time: " + str(totalTime))
    return totalTime

def run(cython=False):
    totalTime = calcPi(3000, 50000)
    with open("data/numpy.dat", "a") as f:
        f.write("%f\n" % totalTime)

if __name__ == "__main__":
    run()
