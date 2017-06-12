"""
Calculation of Pi using a Monte Carlo method.
"""

from math import hypot
from random import random
from time import time
import sys

# A range is used in this function for python3. If you are using python2, a
# xrange might be more efficient.
def test(tries):
    return sum(hypot(random(), random()) < 1 for _ in range(tries))


# Calculates pi with a Monte-Carlo method. This function calls the function
# test "n" times with an argument of "t". Scoop dispatches these 
# functions interactively accross the available ressources.
def calcPi(workers, tries):
    cdef double bt = time()
    cdef double piValue = 4. * sum(map(test, [tries] * workers)) / float(workers * tries)
    cdef double totalTime = time() - bt
    print("pi = " + str(piValue))
    print("total time: " + str(totalTime))
    return totalTime

def run():
    cdef float totalTime = calcPi(3000, 5000)
    with open("data/cython.dat", "a") as f:
        f.write("%f\n" % totalTime)
