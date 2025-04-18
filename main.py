import numpy as np
import matplotlib.pyplot as plt
from benchmark_functions import sphere, rastrigin, ackley, griewank, zakharov, rastrigin_ii
from dynamic_ga_latent import dynamic_ga_latent
import matplotlib
matplotlib.use('Agg')

benchmarks = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Griewank": griewank,
    "Zakharov": zakharov,
    "Rastrigin II": rastrigin_ii
}

results = {}
final_costs = {}
print("\nRunning Latent-Space GA Once Per Benchmark:\n")
for name, fn in benchmarks.items():
    print(f"Running: {name}")
    costs = dynamic_ga_latent(fn)
    results[name] = costs
    final_costs[name] = costs[-1]

# Plotting
plt.figure(figsize=(12, 7))
for name, costs in results.items():
    plt.semilogy(costs, label=name)
plt.xlabel("Iteration")
plt.ylabel("Best Cost (log scale)")
plt.title("Latent-Space GA: Cost vs Iteration")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("plot.png")


# Print final best costs
print("\nFinal Best Costs:")
for name, cost in final_costs.items():
    print(f"{name:15}: {cost:.6e}")
