"""
Calculation of Pi using a Monte Carlo method.
"""

from math import hypot
from random import random
from time import time
import sys
import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t

# A range is used in this function for python3. If you are using python2, a
# xrange might be more efficient.
def test(tries):
    cdef np.ndarray results = np.zeros(tries, dtype=DTYPE)
    for i in range(tries):
        results[i] = hypot(random(), random()) < 1
    return results.sum()

# Calculates pi with a Monte-Carlo method. This function calls the function
# test "n" times with an argument of "t". Scoop dispatches these 
# functions interactively accross the available ressources.
def calcPi(workers, tries):
    cdef double bt = time()
    cdef np.ndarray expr = np.zeros(workers, dtype=DTYPE)
    for i in range(workers):
        expr[i] = test(tries)
    cdef double piValue = 4. * expr.sum() / float(workers * tries)
    cdef double totalTime = time() - bt
    print("pi = " + str(piValue))
    print("total time: " + str(totalTime))
    return totalTime

def run():
    cdef double totalTime = calcPi(3000, 5000)
    with open("data/cython_numpy.dat", "a") as f:
        f.write("%f\n" % totalTime)

