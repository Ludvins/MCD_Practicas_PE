# -*- coding: UTF-8 -*-

"""
Collection of functions for the first batch of exercises.

All functions from the practical parts of each exercise are
gathered in this module.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.stats import poisson, expon, uniform, norm


def simulate_poisson_process_v1(rate, t0, t1, M):
    """Simulate a Poisson process in [t0,t1].

    The simulation is carried out leveraging the fact that interarrival
    times follow an exponential distribution with the same rate.

    Parameters
    ----------
    rate : float
        Arrival rate of events.
    t0 : float
        Initial time for the simulation
    t1 : float
        Final time for the simulation
    M : int
        Number of processes in simulation

    Returns
    ---------
    arrivals : List of numpy.ndarray [float]
        Each element of the list represents the arrival
        times of a different simulation.

    Example
    -------
    >>> rate = 2
    >>> M = 100
    >>> t0, t1 = 0, 1
    >>> arrivals = simulate_poisson_process_v1(rate, t0, t1, M)
    >>> for m, arr in enumerate(arrivals):
    >>>     print(f"Arrival times for process #{m}:")
    >>>     print(arr)
    """
    arrivals = [[] for _ in range(M)]
    beta = 1.0/rate
    for m in range(M):
        s = t0 + expon.rvs(scale=beta)
        while s < t1:
            arrivals[m].append(s)
            s += expon.rvs(scale=beta)
        arrivals[m] = np.array(arrivals[m])

    return arrivals


def simulate_poisson_process_v2(rate, t0, t1, M):
    """Simulate a Poisson process in [t0,t1].

    The simulation is carried out using the order statistics of
    the uniform distribution in [t0, t1].

    Parameters
    ----------
    rate : float
        Arrival rate of events.
    t0 : float
        Initial time for the simulation
    t1 : float
        Final time for the simulation
    M : int
        Number of processes in simulation

    Returns
    ---------
    arrivals : List of numpy.ndarray [float]
        Each element of the list represents the arrival
        times of a different simulation.

    Example
    -------
    >>> rate = 2
    >>> M = 100
    >>> t0, t1 = 0, 1
    >>> arrivals = simulate_poisson_process_v2(rate, t0, t1, M)
    >>> for m, arr in enumerate(arrivals):
    >>>     print(f"Arrival times for process #{m}:")
    >>>     print(arr)
    """
    arrivals = [[] for _ in range(M)]
    for m in range(M):
        # Sample 'n' from a Poisson distribution
        n = poisson.rvs((t1 - t0)*rate)
        # Generate n independen samples from U[t0, t1]
        u = uniform.rvs(loc=t0, scale=t1 - t0, size=n)
        # Order the samples and interpret them as time events
        arrivals[m] = np.sort(u)

    return arrivals


def simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N):
    """Simulate in [t0,t0+T] of arithmetic Brownian motion trajectories.

    The process is given by the expression:

        dB(t) = mu*dt + sigma*dW(t)

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    B0 : float
        Initial level of the process
    T : float
        Length of the simulation
    mu, sigma : float
        Parameters of the process
    M : int
        Number of trajectories in simulation
    N : int
        Number of steps for the simulation

    Returns
    -------
    t: numpy.ndarray [float] of shape (N+1,)
        Regular grid of discretization times in [t0,t0+T]
    B: numpy.ndarray [float] of shape (M, N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values of the process at t

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import BM_simulators as BM
    >>> t0, B0, T, mu, sigma = 0, 10.0, 2.0, 1.5, 0.4
    >>> M, N = 20, 1000
    >>> t, B = BM.simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> _ = plt.plot(t,B.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('B(t)')
    >>> _= plt.title('Arithmetic Brownian motion in 1D')

    Created on Sun Sep 27 19:50:45 2020
    @author: alberto
    """
    dT = T / N    # integration step
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    Z = np.random.randn(M, N)    # Gausssian White noise
    d = mu*dT + sigma*np.sqrt(dT)*Z  # additive factor
    B = np.concatenate((B0*np.ones((M, 1)), d), axis=1)
    B = np.cumsum(B, axis=1)
    return t, B


def plot_trajectories_and_distribution(t, B, t0, B0, T, mu, sigma,
                                       name, max_trajectories=50,
                                       max_bins=50,
                                       plot_expected=True):
    """Plot trajectories of the simulations of a Brownian process.

    It plots the trajectories along with the theoretical distribution
    at the last time step.

    Parameters
    ----------
    t : numpy.ndarray [float] or array-like of shape (N,)
        Regular grid of discretization times in [t0, t0+T]
    B: numpy.ndarray [float] or array-like of shape (M, N)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values of the process at t
    t0 : float
        Initial time for the simulation
    B0 : float
        Initial level of the process
    T : float
        Length of the simulation
    mu, sigma : float
        Parameters of the process
    name : string
        Name of the process
    max_trajectories : int
        Maximum number of trajectories to show in graph
    max_bins : int
        Maximum number of bins for the histogram
    plot_expected : boolean
        Whether to plot the expected surrounding area of the trajectories

    Example
    -------
    >>> M = 10000
    >>> N = 1000
    >>> T = 2.0
    >>> mu, sigma = 0.0, 1.0
    >>> t0, B0 = 0.0, 0.0

    >>> t, W = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> plot_trajectories_and_distribution(t, W, t0, B0, T, mu, sigma, 'B(t)')
    """
    # Plot trajectories
    M, _ = np.shape(B)
    M = np.min((M, max_trajectories))
    plt.plot(t, B[:M, :].T, lw=1)
    plt.xlabel('t')
    plt.ylabel(f"{name}")
    plt.title('Simulation')
    plt.plot(t, np.mean(B, axis=0), lw=3, color='k')

    # Plot expected region
    if plot_expected:
        expected_mean = 0
        expected_std = np.sqrt(t)
        a = 3.0
        plt.fill_between(t, expected_mean - a*expected_std,
                         expected_mean + a*expected_std, alpha=0.3)
    plt.show()

    # Plot distribution at t0 + T
    BT = B[:, -1]
    def pdf(x): return norm.pdf(x, B0 + mu*T, sigma*np.sqrt(T))
    n_bins = np.min((np.int(np.round(np.sqrt(len(BT)))), max_bins))
    plt.hist(BT, bins=n_bins, density=True)
    plt.xlabel('x')
    plt.ylabel(f"pdf({name} = x)")

    # Compare with exact distribution
    n_plot = 1000
    x_plot = np.linspace(np.min(BT), np.max(BT), n_plot)
    y_plot = pdf(x_plot)
    plt.plot(x_plot, y_plot, lw=2, color='r')
    plt.title(f"Distribution at T={t0+T}")
    plt.show()


def animate_arithmetic_BM(t0, B0, T, mu, sigma, M, N, figsize=None, max_trajectories=100):
    """Animate simulation in [t0,t0+T] of arithmetic Brownian motion trajectories.

    The process is given by the expression:

        dB(t) = mu*dt + sigma*dW(t),

    It also plots the empirical distribution at each time.

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    B0 : float
        Initial level of the process
    T : float
        Length of the simulation
    mu, sigma : float
        Parameters of the process
    M : int
        Number of trajectories in simulation
    N : int
        Number of steps for the simulation
    figsize : (float float)
        Width, height in inches. If not provided, defaults to [6.4, 4.8]
    max_trajectories : int
        Maximum number of trajectories to show in graph

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Constructed animation. Can be saved with `save(self, filename)`, and exported
        to HTML with `to_jshtml` or to HTML5 video with `to_html5_video`.

    Example
    -------
    >>> M = 1000
    >>> N = 1000
    >>> T = 2.0
    >>> mu, sigma = 1.3, 0.4
    >>> t0, B0 = 0.0, 10.0

    >>> _ = animate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    Mt = np.min((M, max_trajectories))

    # Simulate M Brownian trajectories
    t, B = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)

    def pdf(x, t):
        """Compute the theoretical pdf at time 't', evaluated in 'x'."""
        if t == t0:
            return np.full_like(x, B0)
        return norm.pdf(x, B0 + mu*(t - t0),
                        sigma*np.sqrt(t - t0))

    def init(t, B, step):
        """Initialize figure labels, titles and limits.

        Also clear any previous plots.
        """
        # Clear outputs
        axes[0].clear()
        axes[1].clear()

        # Set information
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('B(t)')
        axes[0].set_xlim(t[0], t[-1])
        axes[0].set_ylim(np.min(B), np.max(B))
        axes[0].set_title(
            f'Arithmetic Brownian B({mu}, {sigma}) motion in 1D at t = {t[step]:.3f}')
        axes[1].set_title(
            f'Distribution of B({t[step]:.3f}) = x | B({t0}) = {B0}')
        axes[1].set_xlabel('x')

    def update(i):
        """Update output in animation by advancing frames."""
        # Advance 10 timesteps for each call
        step = 10*i

        # Set labels and limits
        init(t, B, step)

        # Plot Brownian up to t=t[step]
        axes[0].plot(t[:step], B.T[:step, :Mt])
        axes[0].plot(t[:step], np.mean(B.T[:step], axis=1), lw=3, color='k')

        # Plot empirical distribution at t=t[step]
        axes[1].hist(B[:, step],
                     density=True,
                     bins=100,
                     label="Empirical distribution")

        # Plot theoretical distribution
        left, right = axes[1].get_xlim()
        x = np.arange(left, right, 0.01)
        y = pdf(x, t[step])
        axes[1].plot(x, y, color="red", lw=2, label="True pdf")

        # Show legend
        axes[1].legend()

    return anim.FuncAnimation(fig, update, frames=int(N / 10) + 1, repeat=False)
