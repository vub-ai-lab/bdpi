import numpy as np

def task(oldstate, a, newstate):
    if (a == 0) and (newstate[1] > 0.5).all():
        return 1.0, None
    else:
        return 0.0, None
