from __future__ import absolute_import, unicode_literals
from .celery import app
from time import sleep


@app.task
def add(x, y):
    sleep(30)
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def xsum(numbers):
    return sum(numbers)
