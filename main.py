import time
import numpy as np
from ga_functions import dynamic_ga_latent


# Run the latent GA 30 times
n_runs = 30
latent_cost = []
latent_time = []

print("\n--- Running Latent-Space GA 30 Times (2D → 5000D) ---\n")
for run in range(n_runs):
    print(f"Run {run+1:2d}...", end='')
    final_cost, total_time = dynamic_ga_latent()
    latent_cost.append(final_cost)
    latent_time.append(total_time)
    print(f" Done: Final Cost = {final_cost:.6f}, Time = {total_time:.2f} sec")

# Summary
print("\n--- Summary of Latent GA Runs ---")
for i in range(n_runs):
    print(f"Run {i+1:2d}: Final Cost = {latent_cost[i]:.6f}, Time = {latent_time[i]:.2f} sec")


