#!/usr/bin/env python3

import numpy as np

def poisson_process(_lambda, t0, t1):
    ts = [t0]
    while ts[-1] < t1:
        ts.append(ts[-1] + np.random.exponential(1/_lambda))

    return ts[1:-1]
