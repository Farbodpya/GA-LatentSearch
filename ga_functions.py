import random
import numpy as np
from cost_function import sphere
from projection import decode
import time

d = 2 

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

# Dynamic GA in latent space
def dynamic_ga_latent(z_dim=d, n_pop_init=500, max_it=10000, pc=0.8, pm=0.3, mu=0.02):
    global NFE
    NFE = 0
    n_pop = n_pop_init
    Time = np.zeros(max_it)

    pop = initialize_population_latent(n_pop, z_dim, -5, 5)
    pop.sort(key=lambda x: x['Cost'])
    best_sol = pop[0]
    best_cost = [best_sol['Cost']]
    worst_cost = pop[-1]['Cost']
    nfe = [NFE]

    for it in range(max_it):
        start_time = time.time()

        # Crossover in latent space
        popc = []
        nc = 2 * round(pc * n_pop / 2)
        for _ in range(nc // 2):
            i1, i2 = random.randint(0, n_pop - 1), random.randint(0, n_pop - 1)
            p1, p2 = pop[i1], pop[i2]
            y1, y2 = blend_crossover(p1['Position'], p2['Position'])
            x1, x2 = decode(y1), decode(y2)
            popc.append({'Position': y1, 'Cost': sphere(x1)})
            popc.append({'Position': y2, 'Cost': sphere(x2)})

        # Mutation in latent space
        popm = []
        nm = round(pm * n_pop)
        for _ in range(nm):
            i = random.randint(0, n_pop - 1)
            p = pop[i]
            y = mutate(p['Position'], mu)
            x = decode(y)
            popm.append({'Position': y, 'Cost': sphere(x)})

        # Merge and select next generation
        pop = pop + popc + popm
        pop.sort(key=lambda x: x['Cost'])
        worst_cost = max(worst_cost, pop[-1]['Cost'])
        pop = pop[:n_pop]

        best_sol = pop[0]
        best_cost.append(best_sol['Cost'])
        nfe.append(NFE)

        # Decrease population size
        n_pop = max(50, int(n_pop * 0.99))

        end_time = time.time()
        Time[it] = end_time - start_time

        # Optional: Early stopping
        if best_cost[-1] < 1e-6:
            break

    total_time = np.sum(Time)
    return best_cost[-1], total_time
