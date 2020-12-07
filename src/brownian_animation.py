#!/usr/bin/env python3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy as sp
import matplotlib.animation as anim
from BM_simulators import simulate_arithmetic_BM


def animate_arithmetic_BM(t0, B0, T, mu, sigma, M, N, figsize=None):
    """ Animated simulation in [t0,t0+T] of arithmetic Brownian motion trajectories

        SDE:    dB(t) = mu*dt + sigma*dW(t)

    and empirical distribution.

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
    figsize: (float float)
        Width, height in inches. If not provided, defaults to [6.4, 4.8].


    Returns
    -------
    matplotlib.animation.FuncAnimation
        Constructed animation. Can be saved with `save(self, filename)` or exported to
        html5 video with `to_html5_video`.

    Example
    -------
    >>>> M = 1000
    >>> N = 1000
    >>> T = 2.0
    >>> mu, sigma = 1.3, 0.4
    >>> t0, B0 = 0.0, 10.0

    >>> a = BA.animate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)
    >>> plt.show()
    """

    # Simulate Brownian using BM_simulatiors function
    t, B = simulate_arithmetic_BM(t0, B0, T, mu, sigma, M, N)

    # Define a function to compute the theoretical distribution from scipy.stats
    def pdf(x, t):
        return sp.stats.norm.pdf(x, B0 + mu * (t - t0),
                                 sigma * np.sqrt(t - t0))

    # Create figure and axes for plottings
    fig, axes = plt.subplots(1, 2, figsize = figsize)

    # Define the two needed functions to use plt.animations function. Init and update.
    def init():
        """
            Initializes each figures label and limits.
        """
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('B(t)')
        axes[0].set_xlim(t[0], t[-1])
        axes[0].set_ylim(np.min(B), np.max(B))

    def update(i):
        """
            Clears previous output, plots the updated one while updating the title.
        """
        # Compute this step in [0,1000]
        step = 10 * i
        # Clear outputs
        axes[0].clear()
        axes[1].clear()
        # Set labels and limits
        init()
        # Set titles
        axes[0].set_title(
            f'Arithmetic Brownian B({mu}, {sigma}) motion in 1D at t={t[step]:.3f}'
        )
        axes[1].set_title(f'Distribution of B({t[step]:.3f}) = x | B({t0}) = {B0}')
        # Plot Brownian up to t=t[step]
        axes[0].plot(t[:step], B.T[:step])
        # Plot empirical distribution at t=t[step]
        axes[1].hist(B[:, step],
                     density=True,
                     bins=100,
                     label="Empirical distribution")

        # Get histogram limits in order to plot the theoretical density
        # in them
        left, right = axes[1].get_xlim()
        x = np.arange(left, right, 0.01)
        y = pdf(x, t[step])
        axes[1].plot(x, y, color="red", lw=2, label="True pdf")

        # Show legend
        axes[1].legend()

    return anim.FuncAnimation(fig, update, frames=int(N / 10), repeat=False)
