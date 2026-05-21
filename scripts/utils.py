import math


def exists(x):
    return x is not None

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def cycle(dl):
    while True:
        for data in dl:
            yield data

def divisible_by(numer, denom):
    return (numer % denom) == 0