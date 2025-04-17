import random
import numpy as np
from cost_function import sphere
from projection import decode

# Blend Crossover (BLX-alpha)
def blend_crossover(x1, x2, alpha=0.5):
    gamma = (1 + 2 * alpha) * np.random.rand(*x1.shape) - alpha
    y1 = (1 - gamma) * x1 + gamma * x2
    y2 = gamma * x1 + (1 - gamma) * x2
    return y1, y2

# Gaussian Mutation
def mutate(x, mu, sigma=0.1):
    nmu = int(np.ceil(mu * len(x)))
    indices = random.sample(range(len(x)), nmu)
    y = np.copy(x)
    for i in indices:
        y[i] += sigma * np.random.randn()
    return y

# Initialize latent-space population
def initialize_population_latent(n_pop, z_dim, var_min=-5, var_max=5):
    pop = []
    for _ in range(n_pop):
        z = np.random.uniform(var_min, var_max, size=z_dim)
        x = decode(z)
        cost = sphere(x)
        pop.append({'Position': z, 'Cost': cost})
    return pop
