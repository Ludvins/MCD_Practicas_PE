import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, erlang
from arrival import simulate_poisson_process_v2

# Process parameters
rate = 5
beta = 1.0 / rate
ns = [1, 2, 5, 10]

# Theoretical PDF
ts = np.arange(0, 6, 0.01)
pdfs = [erlang.pdf(ts, a = n, scale = beta) for n in ns]

# Simulation
M = int(1e4)
events = simulate_poisson_process_v2(rate, 0, ts[-1], M)
events = [[events[m][n - 1] for m in range(M)] for n in ns]
density_estimates = [gaussian_kde(events[n]) for n in range(len(ns))]

# Plot results to compare
fig, axs = plt.subplots(2, 2)

for i in [0, 1]:
    for j in [0, 1]:
        n = 2 * i + j
        axs[i, j].hist(events[n], label = f"Simulated S{ns[n]}", density = True, edgecolor = 'k', bins = 20 + n)
        axs[i, j].plot(ts, pdfs[n], label = f"True density of S{ns[n]}",
                       color = "red", lw = 2)
        axs[i, j].plot(ts, density_estimates[n](ts), "--", label = f"Estimated density of S{ns[n]}", lw = 2, color = "gold")
        axs[i, j].set_title(f"Distribution of event #{ns[n]}")
        axs[i, j].set_xlabel("Time")
        axs[i, j].set_ylabel("Density")
        axs[i, j].legend()
plt.show()
