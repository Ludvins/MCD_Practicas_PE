#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang

l = 10
t = 2

def f(n):
    print(n)
    print(1.0/np.math.factorial(n))
    print(l**n)
    print(t**n)
    print(np.exp(l*t))
    return (1.0/np.math.factorial(n)) * (l**n) * (t**n) / np.exp(l*t)

t0 = 0
t1 = t


beta = 1.0/l

def poisson_process(beta, t0, t1):
    ts = [t0]
    while ts[-1] < t1:
        ts.append(ts[-1] + np.random.exponential(beta))

    return ts[1:-1]

a = poisson_process(beta, 0, 2)
print(a)
count = np.array([len(poisson_process(beta, 0 ,2)) for _ in range(100)])
plt.hist(count, density = True)

n = np.arange(20)
plt.plot(n, [erlang.pdf(x, ) for x in n])

print(n)
print()

plt.show()
