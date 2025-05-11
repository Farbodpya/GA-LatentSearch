import numpy as np
import matplotlib.pyplot as plt
import time
from benchmark_functions import sphere, rastrigin, ackley, griewank, zakharov, rastrigin_ii
from dynamic_ga_latent import dynamic_ga_latent
import matplotlib
matplotlib.use('Agg')

# Define benchmark functions
benchmarks = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Griewank": griewank,
    "Zakharov": zakharov,
    "Rastrigin II": rastrigin_ii
}

# Initialize containers
results = {}
final_costs = {}
runtimes = {}

print("\nRunning Latent-Space GA Once Per Benchmark:\n")

# Run optimizer once per benchmark
for name, fn in benchmarks.items():
    print(f"Running: {name}")
    start = time.time()
    costs = dynamic_ga_latent(fn)
    end = time.time()
    runtime = end - start

    results[name] = costs
    final_costs[name] = costs[-1]
    runtimes[name] = runtime

    print(f"  → Final Cost: {costs[-1]:.6e}")
    print(f"  → Runtime   : {runtime:.2f} seconds")

# Plotting convergence
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

# Print summary
print("\nSummary of Final Best Costs and Runtimes:")
for name in benchmarks:
    print(f"{name:15}: Final Cost = {final_costs[name]:.6e}, Time = {runtimes[name]:.2f}s")
