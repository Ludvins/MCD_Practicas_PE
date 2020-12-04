import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gaussian_kde
from arrival import simulate_poisson_process_v1, simulate_poisson_process_v2

# Process parameters
rate = 10
t = 2

# Theoretical PMF
ns = np.arange(50)
pmf = poisson.pmf(ns, rate * t)

# Simulation
M = int(1e4)
events_v1 = simulate_poisson_process_v1(rate, 0, t, M)
counts_v1 = [len(events_v1[m]) for m in range(M)]
density_estimate_v1 = gaussian_kde(counts_v1)
events_v2 = simulate_poisson_process_v2(rate, 0, t, M)
counts_v2 = [len(events_v2[m]) for m in range(M)]
density_estimate_v2 = gaussian_kde(counts_v2)
counts = [counts_v1, counts_v2]
density_estimates = [density_estimate_v1, density_estimate_v2]

# Plot results to compare
fig, axs = plt.subplots(1, 3, sharey = True)
axs[0].stem(ns, pmf)
axs[0].set_title(r"$P[N(2)=n] = \frac{1}{n!} (20)^n e^{-20}$")
axs[0].set_xlabel("Number of events")
axs[0].set_ylabel("Density")

for i, (count, density_estimate) in enumerate(zip(counts, density_estimates)):
    axs[i + 1].hist(count, density = True, bins = 25, edgecolor = 'k')
    axs[i + 1].plot(ns, poisson.pmf(ns, rate * t), label = "True 'density'",
                color = "red", lw = 2)
    axs[i + 1].plot(ns, density_estimate(ns), "--", label = "Estimated density",
                lw = 2, color = "gold")
    axs[i + 1].set_title(f"Simulation (v{i+1}) "
                + r"of a Poisson process in $[0, 2]$ with rate $\lambda = 10$")
    axs[i + 1].legend()
    axs[i + 1].set_xlabel("Number of events")
plt.show()
