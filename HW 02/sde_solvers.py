# -*- coding: utf-8 -*-
"""
@author: Alberto Suárez
         Luis Antonio Ortega Andrés
         Antonio Coín Castro
"""

import numpy as np

def ode_euler(t0, x0, T, a, N):
    """Numerical integration of an ODE using the Euler scheme.

        x(t0) = x0
        dx(t) = a(t, x(t))*dt    [ODE]

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
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T].
    X: numpy.ndarray of shape (N+1,)
        Vector composed of the values of the approximated
        solution trajectory of the equation at t.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> def a(t, x): return 1.3*x
    >>> t0, x0 = 0.0, 100.0
    >>> T = 2.0
    >>> N = 1000
    >>> t, X = sde.ode_euler(t0, x0, T, a, N)
    >>> plt.plot(t, X)
    """
    dT = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros(N + 1)

    # Initial condition
    X[0] = x0

    # Integration of the ODE
    for n in range(N):
        X[n + 1] = X[n] + a(t[n], X[n])*dT

    return t, X


def euler_maruyana(t0, x0, T, a, b, M, N):
    """Numerical integration of an SDE using the stochastic Euler scheme.

        x(t0) = x0
        dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)    [Itô SDE]

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
        Regular grid of discretization times in [t0, t0+T].
    X: numpy.ndarray of shape (M, N+1)
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
    dT = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)

    for n in range(N):
        dW = np.random.randn(M)
        X[:, n + 1] = (X[:, n] + a(t[n], X[:, n])*dT
            + b(t[n], X[:, n])*np.sqrt(dT)*dW)

    return t, X


def milstein(t0, x0, T, a, b, db_dx, M, N):
    """Numerical integration of an SDE using the stochastic Milstein scheme.

        x(t0) = x0
        dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)    [Itô SDE]

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
        Regular grid of discretization times in [t0, t0+T].
    X: numpy.ndarray of shape (M, N+1)
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
    dT = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)

    for n in range(N):
        dW = np.random.randn(M)
        X[:, n + 1] = (X[:, n] + a(t[n], X[:, n])*dT
            + b(t[n], X[:, n])*np.sqrt(dT)*dW
            + 0.5*b(t[n], X[:, n])*db_dx(t[n], X[:, n])
            * (dW**2 - 1)*dT)

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

    dT = T / N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)

    for n in range(N):

        # print(times_of_jumps)
        jump_mask = np.array([((tau > t[n]) & (tau < t[n + 1]))
                              for tau in times_of_jumps],
                             dtype=object)
        any_jump_mask = np.vectorize(lambda x: x.any())(jump_mask)

        X_prev, t_prev = X[:, n], np.full(M, t[n])

        def update(X_prev, t_prev, m, taus, Ys):
            for (tau, Y) in zip(taus, Ys):
                dW = np.random.randn()
                dT_jump = tau - t_prev[m]
                diffusion = (X_prev[m] + a(t_prev[m], X_prev[m]) * dT_jump +
                             b(t_prev[m], X_prev[m]) * np.sqrt(dT_jump) * dW)
                t_prev[m]= tau
                X_prev[m]= diffusion + c(tau, diffusion) * Y

        for m in np.where(any_jump_mask)[0]:
            update(X_prev, t_prev, m, times_of_jumps[m][jump_mask[m]], sizes_of_jumps[m][jump_mask[m]])

        dW = np.random.randn(M)
        dT_jump = t[n + 1] - t_prev
        X[:, n + 1] = (X_prev + a(t_prev, X_prev) * dT_jump +
                       b(t_prev, X_prev) * np.sqrt(dT_jump) * dW)

    return t, X
