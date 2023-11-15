import sys
import time
import datetime

param_str=sys.argv[1]
x_str, y_str = param_str.split(",")
x=float(x_str)
y=float(y_str)

# Rosenbrock function to minimize
a=1
b=100
score=(x-1)**2 + b*(y-x**2)**2

time.sleep(3)

print(round(score, 3), x, y, datetime.datetime.now())

