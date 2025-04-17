import numpy as np

# Global variable to track the number of function evaluations (NFE)
NFE = 0

# Cost function: Sphere (in 1000D space)
def sphere(x):
    global NFE
    NFE += 1
    return np.sum(np.square(x))
