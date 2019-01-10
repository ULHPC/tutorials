import math

def mean(lst):
    return sum(lst) / len(lst)

def standard_dev(lst):
    m = mean(lst)
    variance = sum([(value - m) ** 2 for value in lst])
    return math.sqrt(variance / len(lst))
