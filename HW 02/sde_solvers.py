# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:29:26 2020

@author: Alberto Suárez
"""
# Load packages
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import stochastic_plots as stoch


def euler_maruyana(t0, x0, T, a, b, M, N):
    """ Numerical integration of an SDE using the stochastic Euler scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t,x(t)) that characterizes the drift term
    b :
        Function b(t,x(t)) that characterizes the diffusion term
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values
        of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> t, S = sde.euler_maruyana(t0, S0, T, a, b, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Euler scheme)')

    """

    t = np.arange(t0, t0 + T, T / (N + 1))
    dt = T / (N + 1)
    X = np.zeros(shape=(M, N + 1))
    X[:, 0] = np.ones(M) * x0
    for n in range(1, N + 1):
        X[:, n] = X[:, n - 1] + a(t[n], X[:, n - 1]) * dt + \
        b(t[n], X[:, n - 1]) * np.sqrt(dt) * np.random.normal(0, 1, M)

    return t, X


def milstein(t0, x0, T, a, b, db_dx, M, N):
    """ Numerical integration of an SDE using the stochastic Milstein scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t, x(t)) that characterizes the drift term
    b :
        Function b(t, x(t)) that characterizes the diffusion term
    db_dx:
        Derivative wrt the second argument of b(t, x)
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> def db_dSt(t, St): return sigma
    >>> t, S = sde.milstein(t0, S0, T, a, b, db_dSt, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Milstein scheme)')

    """

    t = np.arange(t0, t0 + T, T / (N + 1))
    dt = T / (N + 1)
    X = np.zeros(shape=(M, N + 1))
    X[:, 0] = np.ones(M) * x0
    for n in range(1, N + 1):
        X[:, n] = X[:, n - 1] + a(t[n], X[:, n - 1]) * dt + \
            b(t[n], X[:, n - 1]) * np.sqrt(dt) * \
            np.random.normal(0, 1, M) + \
            0.5 * b(t[n], X[:, n - 1]) * db_dx(t[n], X[:, n - 1]) * \
            (np.random.normal(0, 1, M)**2 - 1) * dt

    return t, X


def simulate_jump_process(t0, T, simulator_arrival_times, simulator_jumps, M):
    """ Simulation of jump process

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    T : float
        Length of the simulation interval [t0, t0+T]
    simulator_arrival_times: callable with arguments (t0,T)
        Function that returns a list of M arrays of arrival times in [t0, t0+T]
    simulator_jumps: callable with argument N
        Function that returns a list of M arrays with the sizes of the jumps
    M: int
        Number of trajectories in the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    """

    times_of_jumps = [[] for _ in range(M)]
    sizes_of_jumps = [[] for _ in range(M)]
    for m in range(M):
        times_of_jumps[m] = simulator_arrival_times(t0, T)
        max_jumps = len(times_of_jumps[m])
        sizes_of_jumps[m] = simulator_jumps(max_jumps)
    return times_of_jumps, sizes_of_jumps


# Stochastic Euler scheme for the numerical solution of a jump-diffision SDE
def euler_jump_diffusion(t0, x0, T, a, b, c, simulator_jump_process, M, N):
    """ Simulation of jump diffusion process

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t) + c(t, x(t)) dJ(t)

    [Itô SDE with a jump term]


    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a : Function a(t,x(t)) that characterizes the drift term
    b : Function b(t,x(t)) that characterizes the diffusion term
    c : Function c(t,x(t)) that characterizes the jump term
    simulator_jump_process: Function that returns times and sizes of jumps
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t
    """


    times_of_jumps, sizes_of_jumps = simulator_jump_process(t0, T, M)

    t = np.arange(t0, t0 + T, T / (N + 1))
    dt = T / (N + 1)
    X = np.zeros(shape=(M, N + 1))
    X[:, 0] = np.ones(M) * x0
    for n in range(1, N + 1):

        # print(times_of_jumps)
        jump_mask = np.array([((tau > t[n-1]) & (tau < t[n])) for tau in times_of_jumps], dtype = object)
        any_jump_mask = np.vectorize(lambda x: x.any())(jump_mask)
        filtered_jump_mask = jump_mask[any_jump_mask]

        sizes_of_jumps = np.array(sizes_of_jumps, dtype=object)
        times_of_jumps = np.array(times_of_jumps, dtype = object)

        s = [sizes_of_jumps[any_jump_mask][i][filtered_jump_mask[i]] for i in range(len(filtered_jump_mask))]
        s2 = [times_of_jumps[any_jump_mask][i][filtered_jump_mask[i]] for i in range(len(filtered_jump_mask))]

        print(s)
        print(s2)

        # X[:, n][any_jump_mask] =


        X[:, n] += X[:, n - 1] + a(t[n], X[:, n - 1]) * dt + \
        b(t[n], X[:, n - 1]) * np.sqrt(dt) * np.random.normal(0, 1, M)



    return t, X
