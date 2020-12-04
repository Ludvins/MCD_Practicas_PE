import numpy as np
from scipy.stats import poisson, expon, uniform

def simulate_poisson_process_v1(rate, t0, t1, M):
    """ Perform M independent simulations of a Poisson process
        with rate 'rate', between t0 and t1, leveraging the fact
        that interarrival times ~ exponential(rate)."""

    arrivals = [[] for _ in range(M)]
    for m in range(M):
        ss = [t0 + expon.rvs(scale = 1.0 / rate)]
        while ss[-1] < t1:
            ss.append(ss[-1] + expon.rvs(scale = 1.0 / rate))
        arrivals[m] = ss

    return arrivals

def simulate_poisson_process_v2(rate, t0, t1, M):
    """ Perform M independent simulations of a Poisson process
        with rate 'rate', between t0 and t1, using the order
        statistics of the uniform distribution. """

    arrivals = [[] for _ in range(M)]
    for m in range(M):
        # Sample 'n' from a Poisson distribution
        n = poisson.rvs((t1 - t0) * rate)
        # Generate n independen samples from U[t0, t1]
        u = uniform.rvs(loc = t0, scale = t1 - t0, size = n)
        # Order the samples and interpret them as time events
        arrivals[m] = np.sort(u)

    return arrivals
