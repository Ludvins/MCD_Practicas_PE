import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon, uniform, gaussian_kde

def simulate_poisson_process_v1(rate, t0, t1):
    # Leverage that interarrival times follow an exponential(rate)
    ss = [t0 + expon.rvs(scale = 1.0 / rate)]
    while ss[-1] < t1:
        ss.append(ss[-1] + expon.rvs(scale = 1.0 / rate))

    return ss

def simulate_poisson_process_v2(rate, t0, t1):
    # Sample 'n' from a Poisson distribution
    n = poisson.rvs((t1 - t0) * rate)

    # Generate n independen samples from U[t0, t1]
    u = uniform.rvs(loc = t0, scale = t1 - t0, size = n)

    # Order the samples and interpret them as time events
    return np.sort(u)

# Process parameters
rate = 10
t = 2

# Theoretical PMF
ns = np.arange(50)
pmf = poisson.pmf(ns, rate * t)

# Simulation
max_sims = int(1e4)
events_v1 = np.array([len(simulate_poisson_process_v1(rate, 0, t)) for _ in range(max_sims)])
density_estimate_v1 = gaussian_kde(events_v1)
events_v2 = np.array([len(simulate_poisson_process_v2(rate, 0, t)) for _ in range(max_sims)])
density_estimate_v2 = gaussian_kde(events_v2)

# Plot results to compare
fig, axs = plt.subplots(1, 3, sharey = True)
axs[0].stem(ns, pmf)
axs[0].set_title(r"$P[N(2)=n] = \frac{1}{n!} (20)^n e^{-20}$")

axs[1].hist(events_v1, density = True, bins = 25, edgecolor = 'k')
axs[1].plot(ns, density_estimate_v1(ns), label = "Estimated density")
axs[1].plot(ns, poisson.pmf(ns, rate * t), label = "True 'density'")
axs[1].set_title(r"Simulation (v1) of Poisson process in $[0, 2]$ with rate $\lambda = 10$")
axs[1].legend()

axs[2].hist(events_v2, density = True, bins = 25, edgecolor = 'k')
axs[2].plot(ns, density_estimate_v2(ns), label = "Estimated density")
axs[2].plot(ns, poisson.pmf(ns, rate * t), label = "True 'density'")
axs[2].set_title(r"Simulation (v2) of Poisson process in $[0, 2]$ with rate $\lambda = 10$")
axs[2].legend()
plt.show()
